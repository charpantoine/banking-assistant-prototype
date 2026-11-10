"""Guardrails: Mistral moderation, rule-based prompt-injection check, PII masking for the audit log."""
import re

import config

# Simple phrase rules for a prototype: production would use a dedicated injection classifier.
_INJECTION_PHRASES = [
    "ignore previous instructions", "ignore your instructions", "system prompt", "you are now",
    "ignore tes instructions", "oublie tes instructions", "ignore les instructions",
]

# Applied in this order, so an IBAN is never half-masked as a card or phone number.
_PII_PATTERNS = [
    ("[IBAN]", r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"),
    ("[CARD]", r"\b(?:\d[ -]?){12,18}\d\b"),
    ("[EMAIL]", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    ("[PHONE]", r"(?:\+\d{1,3}[ .-]?|\b0)[1-9](?:[ .-]?\d{2}){4}\b"),
]


def moderate(text):
    """Classify text with Mistral moderation; flagged only on blocking categories, every hit listed."""
    response = config.mistral_client().classifiers.moderate(model=config.MODERATION_MODEL, inputs=[text])
    categories = sorted(name for name, hit in (response.results[0].categories or {}).items() if hit)
    return {"flagged": any(c in config.MODERATION_BLOCKING for c in categories), "categories": categories}


def injection_check(text):
    """Return flagged and the matched rule: a known injection phrase or another customer's id."""
    normalized = " ".join(text.lower().split())
    for phrase in _INJECTION_PHRASES:
        if phrase in normalized:
            return {"flagged": True, "rule": phrase}
    if any(i.upper() != config.CUSTOMER_ID for i in re.findall(r"\bC\d{3}\b", text, re.IGNORECASE)):
        return {"flagged": True, "rule": "other customer id"}
    return {"flagged": False, "rule": None}


def mask_pii(text):
    """Replace IBANs, card numbers, emails and phone numbers with a label such as [IBAN]."""
    for label, pattern in _PII_PATTERNS:
        text = re.sub(pattern, label, text)
    return text
