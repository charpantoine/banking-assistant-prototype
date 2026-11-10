"""Agents: language detection and router (stage 3a); the specialist agents and handle_turn come in 3b."""
import json
import re

import config

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
