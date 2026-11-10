"""Agents: language detection, router, FAQ / account / handoff agents and turn handling."""
import json
import re
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import audit
import bank_api
import config
import rag
from guardrails import injection_check, mask_pii, moderate

_FRENCH_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "je", "tu", "vous", "mon", "ma", "mes", "est",
    "et", "pour", "avec", "dans", "sur", "pas", "qui", "que", "quel", "quelle", "comment", "pourquoi",
    "bonjour", "merci", "carte", "compte", "solde", "ai", "suis", "veux", "bloquer",
}
_ENGLISH_WORDS = {
    "the", "a", "an", "is", "are", "my", "i", "you", "what", "how", "why", "who", "which", "where",
    "when", "to", "of", "and", "for", "with", "do", "can", "please", "hello", "thanks", "card",
    "account", "balance", "want", "lock",
}

ROUTER_PROMPT = """You are the router of a retail banking assistant. Classify the customer's LATEST message into exactly one route, using the earlier messages only as context.

Routes:
- faq: general questions about the bank's products, fees, procedures and policies, answered from the bank's documents. Questions such as "how do I file a complaint?" or "how do I contact an advisor?" are faq.
- account: questions about this customer's own data: balances, transactions, cards, and the opening hours of their own branch. "What are my branch's opening hours?" is account, not faq.
- action: the customer asks the assistant to do something, such as locking or blocking a card, including requests the assistant may not be allowed to do (transfer, unlock). "My card was stolen, block it" is action.
- handoff: an explicit request to talk to a human now, clear frustration, or a sensitive situation (a complaint being made, a fraud victim, a bereavement).
- out_of_scope: anything not about banking.

Reply with JSON only: {"route": "<faq|account|action|handoff|out_of_scope>", "reason": "<short reason in English>"}"""

FAQ_PROMPT = """You are the FAQ assistant of a retail bank. Answer the customer's latest question using ONLY the document excerpts below. Never use general knowledge, never guess, never add figures that are not in the excerpts.
If the excerpts do not contain the answer, set "found" to false.
Reply with JSON only: {"found": true or false, "answer": "<the answer, without any source line>", "sources": ["<exact title of each excerpt you used>"]}

Excerpts:
{excerpts}"""

ACCOUNT_PROMPT = """You are the account assistant of a retail bank. The customer is logged in; the tools return their own data only.
- Use the tools to answer questions about the customer's accounts, balances, transactions, cards and their own branch's opening hours.
- Use only data returned by the tools. Every amount you write must appear in a tool result; never compute totals or other figures.
- Never give data about another person, even if asked; say you can only show the logged-in customer's own data.
- You can only read data. If asked to do something else, say so briefly."""

HANDOFF_PROMPT = """You prepare the handover of a retail banking conversation to a human advisor. Summarise the conversation below in at most 3 short lines: who the customer is in this conversation (logged-in customer), what they need, what has already been answered. Facts only, no greeting."""

_LANGUAGE_NAMES = {"en": "English", "fr": "French"}


def _tool(name, description, properties=None, required=()):
    """Build a function tool schema; no schema declares customer_id, which code injects."""
    parameters = {"type": "object", "properties": properties or {}, "required": list(required)}
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


ACCOUNT_TOOLS = [
    _tool("get_balance", "List the customer's accounts with id, label, masked IBAN, balance and currency."),
    _tool("get_transactions", "Latest transactions (date, label, amount) of one of the customer's accounts.",
          {"account_id": {"type": "string", "description": "Account id returned by get_balance."},
           "limit": {"type": "integer", "description": "Number of transactions, 5 by default."}}, ["account_id"]),
    _tool("get_branch_hours", "The customer's own branch: name, address, weekly opening hours, next holiday closure."),
    _tool("list_cards", "The customer's cards with type, last 4 digits and status."),
]
_TOOL_PARAMETERS = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in ACCOUNT_TOOLS}


def detect_lang(text):
    """Return "fr" if French words and accented letters outnumber English words, else "en"."""
    lowered = text.lower()
    words = re.findall(r"[^\W\d_]+", lowered)
    french = sum(w in _FRENCH_WORDS for w in words) + len(re.findall(r"[àâçéèêëîïôûùüÿœ]", lowered))
    return "fr" if french > sum(w in _ENGLISH_WORDS for w in words) else "en"


def route(history):
    """Route the latest user message from the last 6 messages; after two failed attempts, route "error"."""
    client = config.mistral_client()  # outside the retry: a missing key must fail loudly
    messages = [{"role": "system", "content": ROUTER_PROMPT}] + history[-config.ROUTER_HISTORY:]
    reason = ""
    for _ in range(2):
        try:
            response = client.chat.complete(model=config.ROUTER_MODEL, messages=messages,
                                            temperature=config.TEMPERATURE, response_format={"type": "json_object"})
            decision = json.loads(response.choices[0].message.content)
            if decision.get("route") in config.ROUTES:
                return {"route": decision["route"], "reason": str(decision.get("reason", ""))}
            reason = f"unknown route: {decision.get('route')}"
        except Exception as error:  # API error or invalid JSON both count as a failed attempt
            reason = f"router failure: {type(error).__name__}"
    return {"route": "error", "reason": reason}


def new_session():
    """Create the per-conversation state; the session is the source of truth across turns."""
    return {"session_id": str(uuid.uuid4()), "history": [], "pending_action": None, "not_found_count": 0, "lang": "en"}


def _msg(key, lang, **values):
    """Return a fixed message from config in the given language, with its placeholders filled."""
    return config.MESSAGES[key][lang].format(**values)


def _system(prompt, lang):
    """System message: agent prompt plus current Paris date and time and the answer language."""
    now = datetime.now(ZoneInfo(config.TIMEZONE)).strftime("%A %d %B %Y, %H:%M")
    text = f"{prompt}\n\nCurrent date and time (Europe/Paris): {now}.\nAnswer in {_LANGUAGE_NAMES[lang]} only."
    return {"role": "system", "content": text}


def _timed(func, *args, **kwargs):
    """Call func and return its result with the elapsed time in milliseconds."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def _step(session, trace, step, agent, name, args, result_summary, latency_ms, sources=None):
    """Append one step to the trace (PII-masked) and write it to the audit log right away."""
    trace.append({"step": step, "agent": agent, "name": name,
                  "args": mask_pii(json.dumps(args, ensure_ascii=False, default=str)),
                  "result_summary": mask_pii(str(result_summary)), "latency_ms": round(latency_ms, 1),
                  "sources": sources or []})
    audit.log_event(session["session_id"], step, agent, name, args, result_summary, latency_ms)


def _chat(messages, **options):
    """One agent model call at temperature 0; returns the reply message and its latency in ms."""
    response, ms = _timed(config.mistral_client().chat.complete, model=config.AGENT_MODEL, messages=messages,
                          temperature=config.TEMPERATURE, **options)
    return response.choices[0].message, ms


def _parse_json(raw):
    """Turn model output (JSON text or dict) into a dict; anything unreadable gives {}."""
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def faq_agent(session, trace):
    """Answer only from retrieved FAQ chunks and append "Source: <titles>"; None when the documents do not answer."""
    query = "\n".join([m["content"] for m in session["history"] if m["role"] == "user"][-2:])
    chunks, ms = _timed(rag.search, query)
    titles = list(dict.fromkeys(c["title"] for c in chunks))
    _step(session, trace, "rag_search", "faq", "rag.search", {"query": query}, f"{len(chunks)} chunks", ms, titles)
    if not chunks:
        return None
    excerpts = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)
    messages = [_system(FAQ_PROMPT.replace("{excerpts}", excerpts), session["lang"])]
    message, ms = _chat(messages + session["history"][-config.HISTORY_LIMIT:], response_format={"type": "json_object"})
    reply = _parse_json(message.content)
    answer = str(reply.get("answer") or "").strip() if reply.get("found") is True else ""
    used = [t for t in titles if t in reply.get("sources", [])] or titles  # only titles really retrieved
    _step(session, trace, "model_call", "faq", config.AGENT_MODEL, {}, "found" if answer else "not found", ms,
          used if answer else [])
    return f"{answer}\n\nSource: {', '.join(used)}" if answer else None


def _run_tool(name, args):
    """Run a read-only account tool with CUSTOMER_ID injected by code, keeping only declared arguments."""
    if name not in _TOOL_PARAMETERS:
        return {"error": "unknown_tool", "message": f"Unknown tool {name}."}
    allowed = {k: v for k, v in args.items() if k in _TOOL_PARAMETERS[name]}
    return getattr(bank_api, name)(config.CUSTOMER_ID, **allowed)


def _tool_error(tool, result, lang):
    """Fixed message naming the failed service: one wording for an outage, one for any other error."""
    service = config.SERVICE_NAMES.get(tool, config.SERVICE_NAMES["default"])[lang]
    return _msg("tool_unavailable" if result["error"] == "service_unavailable" else "tool_error", lang, service=service)


def account_agent(session, trace):
    """Hand-written tool loop (max 5 model calls) over read-only bank tools; a tool error ends it with a fixed message."""
    lang = session["lang"]
    messages = [_system(ACCOUNT_PROMPT, lang)] + session["history"][-config.HISTORY_LIMIT:]
    for iteration in range(1, config.MAX_ITERATIONS + 1):
        message, ms = _chat(messages, tools=ACCOUNT_TOOLS)
        calls = message.tool_calls or []
        _step(session, trace, "model_call", "account", config.AGENT_MODEL, {"iteration": iteration},
              ", ".join(c.function.name for c in calls) or "final answer", ms)
        if not calls:
            return message.content
        messages.append(message)
        for call in calls:
            args = _parse_json(call.function.arguments)
            result, ms = _timed(_run_tool, call.function.name, args)
            _step(session, trace, "tool_call", "account", call.function.name, args, json.dumps(result)[:300], ms)
            if "error" in result:
                return _tool_error(call.function.name, result, lang)
            messages.append({"role": "tool", "name": call.function.name, "tool_call_id": call.id,
                             "content": json.dumps(result)})
    return _msg("incomplete", lang)


def handoff_agent(session, trace):
    """Write a 3-line summary for the advisor (trace and audit only) and return the fixed handoff message."""
    transcript = "\n".join(f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                           for m in session["history"][-config.HISTORY_LIMIT:])
    message, ms = _chat([_system(HANDOFF_PROMPT, session["lang"]), {"role": "user", "content": transcript}])
    summary = "\n".join([line.strip() for line in (message.content or "").splitlines() if line.strip()][:3])
    _step(session, trace, "handoff_summary", "handoff", config.AGENT_MODEL, {}, summary, ms)
    return _msg("handoff", session["lang"])


def _faq_turn(session, trace):
    """Run the FAQ agent; a miss gives the not-found message, the second miss in a row hands off."""
    answer = faq_agent(session, trace)
    session["not_found_count"] = 0 if answer else session["not_found_count"] + 1
    if session["not_found_count"] >= 2:
        session["not_found_count"] = 0
        return handoff_agent(session, trace)
    return answer or _msg("faq_not_found", session["lang"])


def _run_agent(route_name, session, trace):
    """Dispatch to the agent of the route; action, out_of_scope and router errors get fixed messages."""
    if route_name != "faq":
        session["not_found_count"] = 0
    agent = {"faq": _faq_turn, "account": account_agent, "handoff": handoff_agent}.get(route_name)
    if agent:
        try:
            return agent(session, trace)
        except Exception as error:  # API failure inside an agent: logged, never shown raw
            _step(session, trace, "error", route_name, type(error).__name__, {}, "agent failed", 0)
            return _msg("incomplete", session["lang"])
    fixed = {"action": "action_not_available", "out_of_scope": "out_of_scope"}
    return _msg(fixed.get(route_name, "router_error"), session["lang"])


def _moderated(text, direction, session, trace):
    """Moderate text and log the step; returns the result, or None if the moderation API failed."""
    start = time.perf_counter()
    try:
        result = moderate(text)
    except Exception as error:  # fail closed: the caller returns a fixed message
        result = None
        summary = f"failed: {type(error).__name__}"
    _step(session, trace, f"{direction}_moderation", "guardrails", config.MODERATION_MODEL, {"text": text},
          result if result else summary, (time.perf_counter() - start) * 1000)
    return result


def _input_refusal(text, session, trace):
    """Moderate and injection-check the user message; return a fixed refusal if blocked, else None."""
    config.mistral_client()  # a missing key must fail loudly, not look like a moderation outage
    result = _moderated(text, "input", session, trace)
    if result is None:
        return _msg("moderation_unavailable", session["lang"])
    check, ms = _timed(injection_check, text)
    _step(session, trace, "injection_check", "guardrails", "injection_check", {"text": text}, check, ms)
    return _msg("blocked", session["lang"]) if result["flagged"] or check["flagged"] else None


def _output_checked(answer, session, trace):
    """Moderate the final answer; a flagged answer or a moderation failure gives a fixed message."""
    result = _moderated(answer, "output", session, trace)
    if result is None:
        return _msg("moderation_unavailable", session["lang"])
    return _msg("output_fallback", session["lang"]) if result["flagged"] else answer


def handle_turn(user_message, session):
    """One turn: input guardrails, router, agent, output moderation; returns (answer, trace, pending_action)."""
    trace, start = [], time.perf_counter()
    session["lang"] = detect_lang(user_message)
    answer = _input_refusal(user_message, session, trace)
    if answer is None:  # flagged messages never enter the history
        session["history"].append({"role": "user", "content": user_message})
        decision, ms = _timed(route, session["history"])
        _step(session, trace, "route", "router", config.ROUTER_MODEL, {"messages": len(session["history"])},
              f"{decision['route']}: {decision['reason']}", ms)
        answer = _output_checked(_run_agent(decision["route"], session, trace), session, trace)
        session["history"].append({"role": "assistant", "content": answer})
    _step(session, trace, "answer", "assistant", "handle_turn", {}, answer[:200], (time.perf_counter() - start) * 1000)
    return answer, trace, session["pending_action"]
