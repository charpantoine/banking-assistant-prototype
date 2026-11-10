"""Tests of the agents: language detection, router, and full turns through handle_turn (API).
Numbered tests follow SPEC.md."""
import json
import re
from types import SimpleNamespace

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
    ("How do I lock my card if I lose it?", "faq"),
    ("What can you do?", "help"),
    ("Comment bloquer ma carte si je la perds ?", "faq"),
]

AMOUNT = r"\d{1,3}(?:[   .,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?"
FRENCH_WORDS = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "est", "votre", "vos", "vous", "pour",
                "en", "sur", "dans", "pas", "jours", "virement", "peut", "être", "je", "ne", "nos", "aux", "puis"}
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


def is_french(text):
    """True if the text has at least 2 words of the test's French list and none of its English list."""
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ]+", text.lower())
    return sum(w in FRENCH_WORDS for w in words) >= 2 and not [w for w in words if w in ENGLISH_WORDS]


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
    """French text gives fr; English and other languages give en; a short message with no signal is unknown."""
    assert agents.detect_lang("Je veux bloquer ma carte") == "fr"
    assert agents.detect_lang("I want to lock my card") == "en"
    assert agents.detect_lang("Quiero bloquear mi tarjeta") == "en"
    assert agents.detect_lang("merci") == "fr"
    assert agents.detect_lang("dis quoi") == "unknown"
    assert agents.detect_lang("ok") == "unknown"


@pytest.mark.parametrize("question, expected", ROUTER_CASES)
def test_router(question, expected):
    """Each question, sent alone, gets its expected route."""
    assert agents.route([{"role": "user", "content": question}])["route"] == expected


def test_01_faq_answer_cites_source():
    """Test 1: a FAQ question is routed to faq and the answer cites a source."""
    answer, trace, _ = ask("How long does a transfer take to arrive?")
    assert route_of(trace) == "faq"
    assert "Source:" in answer
    output = next(s for s in trace if s["step"] == "output_moderation")
    assert output["name"] == config.MODERATION_MODEL  # model-written answer: moderated


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
    discarded = [s for s in trace if s["step"] == "route_discarded"]
    assert len(discarded) == 1 and "discarded: input blocked" in discarded[0]["result_summary"]
    assert any(e["step"] == "injection_check" and "other customer id" in e["result_summary"]
               for e in audit.last_events(10))


def test_07_out_of_scope_refusal():
    """Test 7: a non-banking question gets the fixed out_of_scope refusal, which ends with example questions."""
    answer, trace, _ = ask("Who won the World Cup?")
    assert route_of(trace) == "out_of_scope"
    assert answer == config.MESSAGES["out_of_scope"]["en"]
    assert "For example: ‘What is my balance?’" in answer
    output = next(s for s in trace if s["step"] == "output_moderation")
    assert (output["name"], output["result_summary"]) == ("skipped", "skipped: fixed template")


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


def visitor_session():
    """A new session in visitor mode (not logged in)."""
    return agents.new_session(authenticated=False)


def test_help_after_refusal_routes_to_help_in_french():
    """After an out_of_scope answer, "à quelles questions par exemple" goes to help and gets the French help."""
    session = agents.new_session()
    agents.handle_turn("Qui a gagné la Coupe du monde ?", session)
    answer, trace, _ = agents.handle_turn("à quelles questions par exemple", session)
    assert route_of(trace) == "help"
    assert answer == config.MESSAGES["help"]["fr"]


def test_visitor_balance_gets_login_message_without_bank_call(monkeypatch):
    """Visitor: a balance question gets the login message; no agent model call, no bank tool call."""
    called = []
    monkeypatch.setattr(bank_api, "get_balance", lambda *args, **kwargs: called.append(args) or {})
    session = visitor_session()
    answer, trace, _ = agents.handle_turn("What is my balance?", session)
    assert route_of(trace) == "account"
    assert answer == config.MESSAGES["login_required"]["en"]
    assert called == []
    assert not [s for s in trace if s["step"] in ("model_call", "tool_call")]


def test_visitor_lock_creates_no_pending_action():
    """Visitor: "Lock my debit card" gets the login message and no pending action."""
    session = visitor_session()
    answer, _, pending = agents.handle_turn("Lock my debit card", session)
    assert answer == config.MESSAGES["login_required"]["en"]
    assert pending is None and session["pending_action"] is None
    assert card_status("CARD-1001") == "active"


def test_visitor_faq_still_answers_with_sources_and_no_lock_offer():
    """Visitor: a FAQ question still gets an answer with sources, without the "lock a card now" offer."""
    answer, trace, _ = agents.handle_turn("How do I lock my card if I lose it?", visitor_session())
    assert route_of(trace) == "faq"
    assert "Source:" in answer
    assert config.MESSAGES["lock_offer"]["en"] not in answer


def test_visitor_help_lists_visitor_capabilities_only():
    """Visitor: help lists general questions and handoff only, with the log-in line and FAQ examples."""
    answer, trace, _ = agents.handle_turn("What can you do?", visitor_session())
    assert route_of(trace) == "help"
    assert answer == config.MESSAGES["help_visitor"]["en"]
    assert "Log in to see your accounts" in answer and "balance" not in answer


def test_bank_tool_runner_refuses_without_customer_id(monkeypatch):
    """Second barrier: with no customer id, the tool runner returns an error and calls no bank API."""
    called = []
    monkeypatch.setattr(bank_api, "get_balance", lambda *args, **kwargs: called.append(args) or {})
    assert agents._run_tool("get_balance", {}, None)["error"] == "not_authenticated"
    assert called == []


def test_confirm_as_visitor_executes_nothing():
    """A pending action left in a visitor session is never executed by confirm (no API call)."""
    session = visitor_session()
    session["pending_action"] = {"action": "lock_card", "card_id": "CARD-1001", "card_label": "Debit card ****4821",
                                 "idempotency_key": "key-1"}
    assert agents.confirm_action(session) == config.MESSAGES["login_required"]["en"]
    assert card_status("CARD-1001") == "active"


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
    assert not [s for s in trace if s["step"] in ("route", "route_discarded")]  # router not called
    assert card_status("CARD-1001") == "active"


def test_confirm_refuses_action_outside_allowlist():
    """Even if a pending action is not lock_card, confirm executes nothing and clears it."""
    session = agents.new_session()
    session["pending_action"] = {"action": "transfer", "card_id": "CARD-1001", "card_label": "x",
                                 "idempotency_key": "key-1"}
    assert agents.confirm_action(session) == config.MESSAGES["action_refused"]["en"]
    assert session["pending_action"] is None
    assert card_status("CARD-1001") == "active"


def test_french_confirmation_translates_card_label():
    """A French confirmation names the card in French, without mixing languages (no API call)."""
    session = agents.new_session()
    session["lang"] = "fr"
    session["pending_action"] = {"action": "lock_card", "card_id": "CARD-1001", "card_label": "Debit card ****4821",
                                 "idempotency_key": "key-1"}
    assert agents.confirm_action(session) == "Votre carte de débit ****4821 est maintenant bloquée."


def test_how_to_lock_question_gets_faq_answer_with_offer():
    """A "how do I" question about locking gets a FAQ answer with the fixed lock offer, and no pending action."""
    answer, trace, session = ask("How do I lock my card if I lose it?")
    assert route_of(trace) == "faq"
    assert config.MESSAGES["lock_offer"]["en"] in answer
    assert "Source:" in answer
    assert session["pending_action"] is None


def test_merci_after_handoff_gets_french_courtesy_without_new_handoff():
    """After a handoff, "merci" gets the fixed French courtesy reply: no router, no new handoff summary."""
    session = agents.new_session()
    _, trace, _ = agents.handle_turn("Je veux parler à un conseiller", session)
    assert route_of(trace) == "handoff" and session["handoff_done"] is True
    answer, trace, _ = agents.handle_turn("merci", session)
    assert answer == config.MESSAGES["courtesy"]["fr"]
    assert not [s for s in trace if s["step"] in ("route", "route_discarded", "handoff_summary")]


def test_second_handoff_request_says_already_notified():
    """A second handoff request gets the fixed "already notified" message and no new summary."""
    session = agents.new_session()
    agents.handle_turn("I want to speak to a human", session)
    answer, trace, _ = agents.handle_turn("I want to speak to a human now", session)
    assert route_of(trace) == "handoff"
    assert answer == config.MESSAGES["handoff_already"]["en"]
    assert not [s for s in trace if s["step"] == "handoff_summary"]
    assert agents.new_session()["handoff_done"] is False  # Clear starts a new session


def test_short_unclear_message_keeps_french():
    """In a French conversation, "dis quoi" (no clear language) keeps French and gets a French answer."""
    session = agents.new_session()
    agents.handle_turn("Quel est le solde de mon compte ?", session)
    assert session["lang"] == "fr"
    answer, _, _ = agents.handle_turn("dis quoi", session)
    assert session["lang"] == "fr"
    assert is_french(answer), answer


@pytest.mark.parametrize("authenticated, status", [(True, "logged-in customer"), (False, "visitor, not logged in")])
def test_handoff_prompt_names_customer_status(monkeypatch, authenticated, status):
    """The handoff summary prompt says "logged-in customer" or "visitor" according to the session (no API call)."""
    prompts = []

    def fake_chat(messages, **options):
        """Record the system prompt and return a one-line summary."""
        prompts.append(messages[0]["content"])
        return SimpleNamespace(content="Customer asks for an advisor."), 1.0

    monkeypatch.setattr(agents, "_chat", fake_chat)
    agents.handoff_agent(agents.new_session(authenticated), [])
    assert f"({status})" in prompts[0]
    assert ("logged-in customer" in prompts[0]) is authenticated
