"""Agents: language detection, router, FAQ / account / action / handoff agents, turn handling, confirm and cancel."""
import json
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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

ROUTER_PROMPT = """You are the router of a retail banking assistant. Classify the customer's LATEST message into exactly one route.
The route is decided by the latest user message only. Earlier messages are context only (for example to understand "the credit one"), never a reason to repeat a previous route.

Routes:
- faq: general questions about the bank's products, fees, procedures and policies, answered from the bank's documents. Questions such as "how do I file a complaint?" or "how do I contact an advisor?" are faq. Any question about HOW to do something ("how do I", "how can I", "what should I do if", "comment faire", "comment puis-je", "que faire si") or about a hypothetical situation ("if I lose it", "si je la perds") is faq, even when it is about locking or blocking a card.
- account: questions about this customer's own data: balances, transactions, cards, and the opening hours of their own branch. "What are my branch's opening hours?" is account, not faq.
- action: a direct request for the assistant to act now, such as locking or blocking a card, including requests the assistant may not be allowed to do (transfer, unlock). "My card was stolen, block it" is action.

Contrasting examples:
- "How do I lock my card if I lose it?" -> faq (asks how, hypothetical)
- "Lock my card" / "Bloque ma carte" -> action (direct request to act now)
- handoff: an explicit request to talk to a human now, clear frustration, or a sensitive situation (a complaint being made, a fraud victim, a bereavement).
- out_of_scope: anything not about banking.
- help: questions about the assistant itself: what it can do, what kind of questions to ask, or "for example?" right after a refusal ("what can you do?", "que peux-tu faire ?", "à quelles questions par exemple ?").

Reply with JSON only: {"route": "<faq|account|action|handoff|out_of_scope|help>", "reason": "<short reason in English>"}"""

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

ACTION_PROMPT = """You are the action assistant of a retail bank. The only action you can prepare is a temporary, reversible lock of one of the customer's own cards.
- To lock a card: call list_cards, then propose_lock_card with that card's id. propose_lock_card never locks anything: the customer confirms with a button.
- If the customer named the card (its type, such as "debit" or "credit", or its last 4 digits) and exactly one card matches, propose that card right away, without asking.
- Only if the customer has several cards and the request matches none or more than one, do not propose: ask which card, listing each one by type and last 4 digits.
- For any other action (transfer, payment, unlocking a card, changing a limit, closing an account...), call refuse_action. Never say an action was done."""

HANDOFF_PROMPT = """You prepare the handover of a retail banking conversation to a human advisor. Summarise the conversation below in at most 3 short lines: who the customer is in this conversation ({status}), what they need, what has already been answered. Facts only, no greeting."""

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
ACTION_TOOLS = [
    _tool("list_cards", "The customer's cards with id, type, last 4 digits and status."),
    _tool("propose_lock_card", "Prepare a temporary lock of one of the customer's cards; the customer must confirm it.",
          {"card_id": {"type": "string", "description": "Card id returned by list_cards."},
           "reason": {"type": "string", "description": "Reason in the customer's own words; empty if none given."}}, ["card_id"]),
    _tool("refuse_action", "Refuse any action other than locking a card.",
          {"requested_action": {"type": "string", "description": "The action the customer asked for."}},
          ["requested_action"]),
]
_TOOL_PARAMETERS = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in ACCOUNT_TOOLS}


def detect_lang(text):
    """Return "fr" if French words and accented letters outnumber English words, "en" if not; "unknown" for a
    message of fewer than 3 words with no clear signal (tie), so the caller keeps the session language."""
    lowered = text.lower()
    words = re.findall(r"[^\W\d_]+", lowered)
    french = sum(w in _FRENCH_WORDS for w in words) + len(re.findall(r"[àâçéèêëîïôûùüÿœ]", lowered))
    english = sum(w in _ENGLISH_WORDS for w in words)
    if len(words) < 3 and french == english:
        return "unknown"
    return "fr" if french > english else "en"


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


def new_session(authenticated=True):
    """Create the per-conversation state, logged in by default; the session is the source of truth across turns."""
    return {"session_id": str(uuid.uuid4()), "history": [], "pending_action": None, "not_found_count": 0,
            "lang": "en", "handoff_done": False, "authenticated": authenticated}


def _customer_id(session):
    """The customer id for bank calls: CUSTOMER_ID when logged in, None for a visitor (never C001)."""
    return config.CUSTOMER_ID if session["authenticated"] else None


def _mode_key(key, session):
    """Template key for the session mode: e.g. "help" when logged in, "help_visitor" for a visitor."""
    return key if session["authenticated"] else f"{key}_visitor"


def switch_mode(session, authenticated):
    """Logged in <-> visitor starts a new session (new id, empty history, no pending action), so nothing from the
    other mode carries over; only the language is kept. The audit event records both session ids."""
    new = new_session(bool(authenticated))
    new["lang"] = session["lang"]
    _step(new, [], "mode_switch", "session", "switch_mode", {"from_session": session["session_id"],
          "to_session": new["session_id"], "authenticated": new["authenticated"]}, "new session", 0)
    return new


class FixedText(str):
    """A str written by code from a config template; output moderation skips it (no model text inside)."""


def _msg(key, lang, **values):
    """Return a fixed message from config in the given language, placeholders filled, marked as FixedText."""
    return FixedText(config.MESSAGES[key][lang].format(**values))


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
    """Answer only from retrieved FAQ chunks and append "Source: <titles>"; None when the documents do not answer.
    If a card lock/loss document was used, a fixed line offers to lock a card now."""
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
    if not answer:
        return None
    offer = session["authenticated"] and any(c["source"] in config.CARD_LOCK_DOCS for c in chunks if c["title"] in used)
    offer_line = f"\n\n{_msg('lock_offer', session['lang'])}" if offer else ""
    return f"{answer}{offer_line}\n\nSource: {', '.join(used)}"


def _run_tool(name, args, customer_id):
    """Run a read-only account tool with the customer id injected by code, keeping only declared arguments.
    Second barrier: with no customer id (visitor), no bank call is made."""
    if customer_id is None:
        return {"error": "not_authenticated", "message": "No logged-in customer."}
    if name not in _TOOL_PARAMETERS:
        return {"error": "unknown_tool", "message": f"Unknown tool {name}."}
    allowed = {k: v for k, v in args.items() if k in _TOOL_PARAMETERS[name]}
    return getattr(bank_api, name)(customer_id, **allowed)


def _tool_error(tool, result, lang):
    """Fixed message naming the failed service: one wording for an outage, one for any other error."""
    service = config.SERVICE_NAMES.get(tool, config.SERVICE_NAMES["default"])[lang]
    return _msg("tool_unavailable" if result["error"] == "service_unavailable" else "tool_error", lang, service=service)


def _tool_loop(session, trace, agent, prompt, tools, handle_call):
    """Hand-written function-calling loop (max 5 model calls); handle_call(name, args, session) returns
    (tool result, final answer or None), and a final answer ends the turn without calling the model again."""
    messages = [_system(prompt, session["lang"])] + session["history"][-config.HISTORY_LIMIT:]
    for iteration in range(1, config.MAX_ITERATIONS + 1):
        message, ms = _chat(messages, tools=tools)
        calls = message.tool_calls or []
        _step(session, trace, "model_call", agent, config.AGENT_MODEL, {"iteration": iteration},
              ", ".join(c.function.name for c in calls) or "final answer", ms)
        if not calls:
            return message.content
        messages.append(message)
        for call in calls:
            args = _parse_json(call.function.arguments)
            (result, final), ms = _timed(handle_call, call.function.name, args, session)
            _step(session, trace, "tool_call", agent, call.function.name, args, json.dumps(result)[:300], ms)
            if final:
                return final
            messages.append({"role": "tool", "name": call.function.name, "tool_call_id": call.id,
                             "content": json.dumps(result)})
    return _msg("incomplete", session["lang"])


def _account_call(name, args, session):
    """Run a read-only account tool; any error ends the turn with the fixed message naming the service."""
    result = _run_tool(name, args, _customer_id(session))
    return result, _tool_error(name, result, session["lang"]) if "error" in result else None


def account_agent(session, trace):
    """Account agent: tool loop over the read-only tools get_balance, get_transactions, get_branch_hours, list_cards."""
    return _tool_loop(session, trace, "account", ACCOUNT_PROMPT, ACCOUNT_TOOLS, _account_call)


def _card_in(lang, card_label):
    """Card label in the answer language: the English card type of the mock data is translated for French."""
    card_type, _, digits = card_label.rpartition(" ")
    return f"{config.CARD_TYPES.get(card_type, {}).get(lang, card_type)} {digits}"


def _propose_lock_card(card_id, session):
    """Check in code that the card is the customer's and active, then store the pending lock; nothing is executed."""
    lang, cards = session["lang"], _run_tool("list_cards", {}, _customer_id(session))
    if "error" in cards:
        return cards, _tool_error("propose_lock_card", cards, lang)
    card = next((c for c in cards["cards"] if c["id"] == card_id), None)
    if card is None:
        return {"error": "not_allowed"}, _msg("card_not_found", lang)
    label = f"{card['type']} ****{card['last_4']}"
    if card["status"] != "active":
        return {"error": "already_locked"}, _msg("card_already_locked", lang, card=_card_in(lang, label))
    session["pending_action"] = {"action": "lock_card", "card_id": card["id"], "card_label": label,
                                 "idempotency_key": str(uuid.uuid4())}
    return session["pending_action"], _msg("lock_proposed", lang, card=_card_in(lang, label))


def _action_call(name, args, session):
    """Action tools: list_cards reads, propose_lock_card only proposes, refuse_action and unknown tools refuse."""
    if name == "list_cards":
        result = _run_tool(name, args, _customer_id(session))
        return result, _tool_error(name, result, session["lang"]) if "error" in result else None
    if name == "propose_lock_card":
        return _propose_lock_card(args.get("card_id"), session)
    return {"refused": str(args.get("requested_action", name))}, _msg("action_refused", session["lang"])


def action_agent(session, trace):
    """Action agent: can only propose a card lock (confirmed later by the button) or refuse."""
    return _tool_loop(session, trace, "action", ACTION_PROMPT, ACTION_TOOLS, _action_call)


def handoff_agent(session, trace):
    """Write a 3-line summary for the advisor (trace and audit only) and return the fixed handoff message."""
    transcript = "\n".join(f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                           for m in session["history"][-config.HISTORY_LIMIT:])
    status = "logged-in customer" if session["authenticated"] else "visitor, not logged in"
    prompt = HANDOFF_PROMPT.replace("{status}", status)
    message, ms = _chat([_system(prompt, session["lang"]), {"role": "user", "content": transcript}])
    summary = "\n".join([line.strip() for line in (message.content or "").splitlines() if line.strip()][:3])
    _step(session, trace, "handoff_summary", "handoff", config.AGENT_MODEL, {}, summary, ms)
    return _msg("handoff", session["lang"])


def _handoff_turn(session, trace):
    """Hand off once per session; later handoffs get the fixed "already notified" message and no new summary."""
    if session["handoff_done"]:
        _step(session, trace, "handoff_skipped", "handoff", "handoff_done", {}, "already notified, no new summary", 0)
        return _msg("handoff_already", session["lang"])
    answer = handoff_agent(session, trace)
    session["handoff_done"] = True  # set after the summary, so a failed call can be retried
    return answer


def _faq_turn(session, trace):
    """Run the FAQ agent; a miss gives the not-found message, the second miss in a row hands off."""
    answer = faq_agent(session, trace)
    session["not_found_count"] = 0 if answer else session["not_found_count"] + 1
    if session["not_found_count"] >= 2:
        session["not_found_count"] = 0
        return _handoff_turn(session, trace)
    return answer or _msg("faq_not_found", session["lang"])


def _run_agent(route_name, session, trace):
    """Dispatch to the agent of the route. Visitors get the login message for account and action;
    help, out_of_scope and router errors get fixed messages for the session mode."""
    if route_name != "faq":
        session["not_found_count"] = 0
    if route_name in ("account", "action") and not session["authenticated"]:  # enforced in code, not in a prompt
        _step(session, trace, "auth_check", "session", route_name, {}, "visitor: no agent, no bank call", 0)
        return _msg("login_required", session["lang"])
    if route_name in ("help", "out_of_scope"):
        return _msg(_mode_key(route_name, session), session["lang"])
    agents = {"faq": _faq_turn, "account": account_agent, "action": action_agent, "handoff": _handoff_turn}
    agent = agents.get(route_name)
    if agent:
        try:
            return agent(session, trace)
        except Exception as error:  # API failure inside an agent: logged, never shown raw
            _step(session, trace, "error", route_name, type(error).__name__, {}, "agent failed", 0)
            return _msg("incomplete", session["lang"])
    return _msg("router_error", session["lang"])


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
    """Moderate the final answer; a flagged answer or a moderation failure gives a fixed message.
    A fixed template from config is code-written text: moderation is skipped and the skip is logged."""
    if isinstance(answer, FixedText):
        _step(session, trace, "output_moderation", "guardrails", "skipped", {}, "skipped: fixed template", 0)
        return answer
    result = _moderated(answer, "output", session, trace)
    if result is None:
        return _msg("moderation_unavailable", session["lang"])
    return _msg("output_fallback", session["lang"]) if result["flagged"] else answer


def _local_reply_key(text, session):
    """Template key for a message answered without the router, else None: a typed "yes" while an action is
    pending (never executes: use the button), then a courtesy word ("thanks", "merci"...)."""
    words = " ".join(text.lower().replace("\u2019", "'").strip(" .!?").split())
    if session["pending_action"] is not None and words in config.CONFIRM_WORDS:
        return "use_confirm_button"
    return "courtesy" if words in config.COURTESY_WORDS else None


def _log_route(decision, ms, discarded, session, trace):
    """Log the router decision; when it is not used (input blocked) log it as discarded."""
    summary = f"{decision['route']}: {decision['reason']}"
    if discarded:
        _step(session, trace, "route_discarded", "router", config.ROUTER_MODEL, {}, f"{summary} (discarded: {discarded})", ms)
    else:
        _step(session, trace, "route", "router", config.ROUTER_MODEL, {}, summary, ms)


def handle_turn(user_message, session):
    """One turn: input guardrails, in parallel with the router unless a local reply (typed "yes", courtesy word)
    needs no router; then agent and output moderation. A flagged message discards the route.
    Returns (answer, trace, pending_action); the session is the source of truth."""
    trace, start = [], time.perf_counter()
    lang = detect_lang(user_message)
    session["lang"] = session["lang"] if lang == "unknown" else lang  # switch only on a clear detection
    history = session["history"] + [{"role": "user", "content": user_message}]
    local = _local_reply_key(user_message, session)
    with ThreadPoolExecutor(max_workers=1) as pool:
        routing = None if local else pool.submit(_timed, route, history)  # runs while the guardrails run
        answer = _input_refusal(user_message, session, trace)
        decision, route_ms = routing.result() if routing else (None, 0)
    if answer is not None:  # flagged: route discarded, message never enters the history
        if routing:
            _log_route(decision, route_ms, "input blocked", session, trace)
    else:
        session["history"].append(history[-1])
        if local:
            _step(session, trace, "local_reply", "guardrails", local, {"text": user_message}, "router not called", 0)
            answer = _msg(local, session["lang"])
        else:
            _log_route(decision, route_ms, None, session, trace)
            answer = _output_checked(_run_agent(decision["route"], session, trace), session, trace)
        session["history"].append({"role": "assistant", "content": answer})
    _step(session, trace, "answer", "assistant", "handle_turn", {}, answer[:200], (time.perf_counter() - start) * 1000)
    return answer, trace, session["pending_action"]


def _closing_message(session, step, args, answer):
    """Log a confirm or cancel step, add the message to the history and return it."""
    _step(session, [], step, "action", step, args, answer, 0)
    session["history"].append({"role": "assistant", "content": answer})
    return answer


def confirm_action(session):
    """Execute the pending action with the customer id and its idempotency key, if logged in and allowlisted.
    An outage keeps the pending action so a retry reuses the same key; any other result clears it."""
    lang, pending, customer_id = session["lang"], session["pending_action"], _customer_id(session)
    if pending is None:
        return _closing_message(session, "confirm", {}, _msg("no_pending_action", lang))
    if customer_id is None:  # visitor: nothing is executed
        session["pending_action"] = None
        return _closing_message(session, "confirm", pending, _msg("login_required", lang))
    if pending["action"] not in config.ALLOWED_ACTIONS:
        session["pending_action"] = None
        return _closing_message(session, "confirm", pending, _msg("action_refused", lang))
    result, ms = _timed(bank_api.lock_card, customer_id, pending["card_id"], pending["idempotency_key"])
    _step(session, [], "action", "action", "lock_card", pending, json.dumps(result), ms)
    if result.get("error") != "service_unavailable":
        session["pending_action"] = None
    done = _msg("lock_done", lang, card=_card_in(lang, pending["card_label"]))
    answer = _tool_error("lock_card", result, lang) if "error" in result else done
    return _closing_message(session, "confirm", pending, answer)


def cancel_action(session):
    """Clear the pending action without executing anything; returns the message."""
    pending, session["pending_action"] = session["pending_action"], None
    answer = _msg("action_cancelled" if pending else "no_pending_action", session["lang"])
    return _closing_message(session, "cancel", pending or {}, answer)
