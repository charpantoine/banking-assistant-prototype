"""Tests of the agents. Stage 3a: language detection and the router (API)."""
import pytest

import agents

ROUTER_CASES = [
    ("How long does a transfer take to arrive?", "faq"),
    ("What is my balance?", "account"),
    ("Lock my debit card", "action"),
    ("I want to speak to a human", "handoff"),
    ("Qui a gagné la Coupe du monde ?", "out_of_scope"),
    ("What are my branch's opening hours?", "account"),
]


def test_detect_lang():
    """French text gives fr; English and other languages give en."""
    assert agents.detect_lang("Je veux bloquer ma carte") == "fr"
    assert agents.detect_lang("I want to lock my card") == "en"
    assert agents.detect_lang("Quiero bloquear mi tarjeta") == "en"


@pytest.mark.parametrize("question, expected", ROUTER_CASES)
def test_router(question, expected):
    """Each question, sent alone, gets its expected route."""
    assert agents.route([{"role": "user", "content": question}])["route"] == expected
