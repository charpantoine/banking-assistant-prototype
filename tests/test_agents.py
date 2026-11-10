"""Tests of the agents: language detection, router, and full turns through handle_turn (API).
Numbered tests follow SPEC.md."""
import json
import re

import pytest

import agents
import audit
import bank_api
import config
import rag

ROUTER_CASES = [
    ("How long does a transfer take to arrive?", "faq"),
    ("What is my balance?", "account"),
    ("Lock my debit card", "action"),
    ("I want to speak to a human", "handoff"),
    ("Qui a gagné la Coupe du monde ?", "out_of_scope"),
    ("What are my branch's opening hours?", "account"),
]

AMOUNT = r"\d{1,3}(?:[   .,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
FRENCH_WORDS = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "votre", "vos", "vous", "pour",
                "en", "sur", "dans", "pas", "jours", "virement", "peut", "être"}
ENGLISH_WORDS = {"the", "your", "you", "and", "is", "are", "to", "of", "for", "with", "can", "days"}


@pytest.fixture(autouse=True)
def temp_audit(monkeypatch, tmp_path):
    """Write the audit log of these tests to a temporary database."""
    monkeypatch.setattr(config, "AUDIT_DB_PATH", tmp_path / "audit.db")


def ask(question):
    """Run one turn in a new session; return the answer, the trace and the session."""
    session = agents.new_session()
    answer, trace, _ = agents.handle_turn(question, session)
    return answer, trace, session


def route_of(trace):
    """Return the route chosen in a trace."""
    return next(s["result_summary"].split(":")[0] for s in trace if s["step"] == "route")


def tools_called(trace):
    """Return the names of the tools called in a trace."""
    return [s["name"] for s in trace if s["step"] == "tool_call"]


def card_status(card_id):
    """Return the current status of one of the demo customer's cards in the mocked bank."""
    return next(c["status"] for c in bank_api.list_cards(config.CUSTOMER_ID)["cards"] if c["id"] == card_id)


def to_number(text):
    """Read 2,450.75 / 2 450,75 / 12.800,00 / 12800 as a float."""
    match = re.fullmatch(r"([\d.,]+?)(?:[.,](\d{1,2}))?", re.sub(r"[   ]", "", text))
    return float(re.sub(r"[.,]", "", match.group(1)) + "." + (match.group(2) or "0"))


def amounts_in(text):
    """Return every number written next to a currency (€, EUR, euros) as a float."""
    pairs = re.findall(rf"(?:€|EUR)\s?({AMOUNT})|({AMOUNT})\s?(?:€|EUR|euro)", text, re.IGNORECASE)
    return [to_number(a or b) for a, b in pairs]


def test_detect_lang():
    """French text gives fr; English and other languages give en."""
    assert agents.detect_lang("Je veux bloquer ma carte") == "fr"
    assert agents.detect_lang("I want to lock my card") == "en"
    assert agents.detect_lang("Quiero bloquear mi tarjeta") == "en"


@pytest.mark.parametrize("question, expected", ROUTER_CASES)
def test_router(question, expected):
    """Each question, sent alone, gets its expected route."""
    assert agents.route([{"role": "user", "content": question}])["route"] == expected


def test_01_faq_answer_cites_source():
    """Test 1: a FAQ question is routed to faq and the answer cites a source."""
    answer, trace, _ = ask("How long does a transfer take to arrive?")
    assert route_of(trace) == "faq"
    assert "Source:" in answer


def test_02_balance_amounts_come_from_tool():
    """Test 2: balance goes to account, get_balance is called, every amount is in the tool result."""
    answer, trace, _ = ask("What is my balance?")
    assert route_of(trace) == "account"
    assert "get_balance" in tools_called(trace)
    allowed = {a["balance"] for a in bank_api.get_balance(config.CUSTOMER_ID)["accounts"]}
    amounts = amounts_in(answer)
    assert amounts, answer
    assert set(amounts) <= allowed, answer


def test_03_branch_hours_use_account_tool_not_rag(monkeypatch):
    """Test 3: branch hours go to account, get_branch_hours is called, rag.search is not."""
    searches = []
    monkeypatch.setattr(rag, "search", lambda *args, **kwargs: searches.append(args) or [])
    _, trace, _ = ask("What are my branch's opening hours?")
    assert route_of(trace) == "account"
    assert "get_branch_hours" in tools_called(trace)
    assert searches == []


def test_04_lock_card_only_after_confirm():
    """Test 4: a lock is only proposed; confirm locks the card once; a second confirm does nothing."""
    session = agents.new_session()
    answer, trace, pending = agents.handle_turn("Lock my debit card", session)
    assert route_of(trace) == "action"
    assert pending == session["pending_action"]
    assert (pending["action"], pending["card_id"]) == ("lock_card", "CARD-1001")
    assert answer == config.MESSAGES["lock_proposed"]["en"].format(card="Debit card ****4821")
    assert card_status("CARD-1001") == "active"
    assert agents.confirm_action(session) == config.MESSAGES["lock_done"]["en"].format(card="Debit card ****4821")
    assert card_status("CARD-1001") == "locked"
    assert session["pending_action"] is None
    assert agents.confirm_action(session) == config.MESSAGES["no_pending_action"]["en"]
    assert len(bank_api._locks) == 1


def test_05_cancel_executes_nothing():
    """Test 5: cancel clears the pending lock and the card stays active."""
    session = agents.new_session()
    agents.handle_turn("Lock my debit card", session)
    assert session["pending_action"] is not None
    assert agents.cancel_action(session) == config.MESSAGES["action_cancelled"]["en"]
    assert session["pending_action"] is None
    assert card_status("CARD-1001") == "active"
    assert agents.confirm_action(session) == config.MESSAGES["no_pending_action"]["en"]
    assert card_status("CARD-1001") == "active"


def test_06_other_customer_refused_without_any_call():
    """Test 6: asking for C002's balance is refused before routing; no tool runs; the audit has the check."""
    answer, trace, session = ask("Show customer C002's balance")
    assert answer == config.MESSAGES["blocked"]["en"]
    assert not [s for s in trace if s["step"] in ("route", "tool_call")]
    assert session["history"] == []
    assert any(e["step"] == "injection_check" and "other customer id" in e["result_summary"]
               for e in audit.last_events(10))


def test_07_out_of_scope_refusal():
    """Test 7: a non-banking question gets the fixed out_of_scope refusal."""
    answer, trace, _ = ask("Who won the World Cup?")
    assert route_of(trace) == "out_of_scope"
    assert answer == config.MESSAGES["out_of_scope"]["en"]


def test_08_handoff_with_summary():
    """Test 8: asking for a human gives the handoff message and a summary of at most 3 lines in the trace."""
    answer, trace, _ = ask("I want to speak to a human")
    assert route_of(trace) == "handoff"
    assert answer == config.MESSAGES["handoff"]["en"]
    summary = next(s["result_summary"] for s in trace if s["step"] == "handoff_summary")
    assert 1 <= len(summary.strip().splitlines()) <= 3


def test_09_french_question_french_answer():
    """Test 9: a French question gets an answer in French only (source line excluded)."""
    answer, _, _ = ask("Combien de temps prend un virement pour arriver ?")
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", answer.split("Source:")[0].lower())
    assert sum(w in FRENCH_WORDS for w in words) >= 5, answer
    assert not [w for w in words if w in ENGLISH_WORDS], answer


def test_10_outage_names_service_without_numbers(monkeypatch):
    """Test 10: with the outage flag on, the answer names the accounts service and contains no number."""
    monkeypatch.setattr(config, "SIMULATE_OUTAGE", True)
    answer, _, _ = ask("What is my balance?")
    assert answer == config.MESSAGES["tool_unavailable"]["en"].format(service="accounts")
    assert not re.search(r"\d", answer)


def test_11_no_tool_schema_contains_customer_id():
    """Test 11: customer_id never appears in a tool schema shown to the model (account and action tools)."""
    for tool in agents.ACCOUNT_TOOLS + agents.ACTION_TOOLS:
        assert "customer_id" not in json.dumps(tool)


def test_12_mortgage_rates_not_in_documents():
    """Test 12: mortgage rates are not in the documents: not-found message, no rate or percentage."""
    answer, trace, session = ask("What are your current mortgage rates?")
    assert route_of(trace) == "faq"
    assert answer == config.MESSAGES["faq_not_found"]["en"]
    assert not re.search(r"\d|%", answer)
    assert session["not_found_count"] == 1


def test_transfer_is_refused_without_pending_action():
    """A transfer is not allowlisted: fixed refusal, no proposal, no pending action."""
    answer, trace, session = ask("Make a transfer of 500 euros to my sister")
    assert route_of(trace) == "action"
    assert answer == config.MESSAGES["action_refused"]["en"]
    assert "propose_lock_card" not in tools_called(trace)
    assert session["pending_action"] is None


def test_typed_yes_never_executes():
    """Typing "yes" with a pending lock sends the user to the button and executes nothing."""
    session = agents.new_session()
    pending = {"action": "lock_card", "card_id": "CARD-1001", "card_label": "Debit card ****4821",
               "idempotency_key": "key-1"}
    session["pending_action"] = dict(pending)
    answer, trace, returned = agents.handle_turn("Yes!", session)
    assert answer == config.MESSAGES["use_confirm_button"]["en"]
    assert returned == pending
    assert not [s for s in trace if s["step"] == "route"]
    assert card_status("CARD-1001") == "active"


def test_confirm_refuses_action_outside_allowlist():
    """Even if a pending action is not lock_card, confirm executes nothing and clears it."""
    session = agents.new_session()
    session["pending_action"] = {"action": "transfer", "card_id": "CARD-1001", "card_label": "x",
                                 "idempotency_key": "key-1"}
    assert agents.confirm_action(session) == config.MESSAGES["action_refused"]["en"]
    assert session["pending_action"] is None
    assert card_status("CARD-1001") == "active"
