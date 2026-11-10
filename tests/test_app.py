"""Tests of the Gradio handlers called directly (no browser). Two of them use the Mistral API."""
import pytest

import app
import bank_api
import config


@pytest.fixture(autouse=True)
def temp_audit(monkeypatch, tmp_path):
    """Write the audit log of these tests to a temporary database."""
    monkeypatch.setattr(config, "AUDIT_DB_PATH", tmp_path / "audit.db")


def test_show_question_returns_placeholder_and_disables_buttons():
    """Step 1 (no API): question and "…" shown, box cleared and disabled, question kept, buttons disabled,
    state untouched; step 3 re-enables the box and the buttons."""
    state = app.new_state()
    chat, box, question, *buttons = app.show_question("What is my balance?", state)
    assert chat == app.welcome_chat("en") + [{"role": "user", "content": "What is my balance?"},
                    {"role": "assistant", "content": config.THINKING_PLACEHOLDER}]
    assert box == {"__type__": "update", "value": "", "interactive": False}
    assert question == "What is my balance?"
    assert len(buttons) == 1 + len(config.EXAMPLE_QUESTIONS)
    assert all(b["interactive"] is False for b in buttons)
    assert state["chat"] == app.welcome_chat("en")
    unlocked = app.unlock_inputs()
    assert len(unlocked) == 2 + len(config.EXAMPLE_QUESTIONS)  # box, Send, examples
    assert all(u["interactive"] is True for u in unlocked)


def test_question_returns_answer_and_trace():
    """Steps 1 then 2: the placeholder is replaced by a non-empty answer, with a trace and no pending card."""
    state = app.new_state()
    _, _, question, *_ = app.show_question("What is my balance?", state)
    chat, state, panel, raw, card, card_text, rows = app.handle_message(question, state)
    assert chat[-2] == {"role": "user", "content": "What is my balance?"}
    assert chat[-1]["role"] == "assistant" and chat[-1]["content"].strip()
    assert config.THINKING_PLACEHOLDER not in [m["content"] for m in chat]
    assert "**Route:** account" in panel and "get_balance" in panel
    assert any(step["name"] == "get_balance" for step in raw)
    assert card["visible"] is False
    assert rows and rows[0][1] == "answer"


def test_action_question_shows_pending_card_then_confirm():
    """An action question shows the Confirm card; Confirm locks the card in code and hides the card."""
    chat, state, _, _, card, card_text, _ = app.handle_message("Lock my debit card", app.new_state())
    assert card["visible"] is True
    assert "Debit card ****4821" in card_text
    key = state["session"]["pending_action"]["idempotency_key"]
    chat, state, panel, raw, card, _, _ = app.handle_confirm(state)
    assert chat[-1]["content"] == config.MESSAGES["lock_done"]["en"].format(card="Debit card ****4821")
    assert card["visible"] is False
    assert "lock_card →" in panel
    assert f"`{key[:8]}…`" in panel and key[8:] not in panel
    assert "executed by code after customer confirmation" in panel
    assert raw is None
    assert [m["role"] for m in chat] == ["assistant", "user", "assistant", "assistant"]  # welcome first; result separate
    assert bank_api.list_cards(config.CUSTOMER_ID)["cards"][0]["status"] == "locked"


def test_clear_starts_new_session_without_pending_action():
    """Clear conversation gives a new session_id, an empty chat and a hidden card (no API call)."""
    state = app.new_state()
    state["session"]["pending_action"] = {"action": "lock_card"}
    chat, box, cleared, panel, raw, card, _, _ = app.handle_clear(config.MODE_CUSTOMER)
    assert chat == app.welcome_chat("en") and box == "" and panel == "" and raw is None
    assert cleared["session"]["session_id"] != state["session"]["session_id"]
    assert cleared["session"]["pending_action"] is None
    assert card["visible"] is False


def test_app_builds_and_outage_switch_sets_flag(monkeypatch):
    """The page builds without launching, and the outage checkbox handler sets the runtime flag."""
    monkeypatch.setattr(config, "SIMULATE_OUTAGE", False)
    assert app.build_app() is not None
    assert app.make_theme() is not None
    app.set_outage(True)
    assert config.SIMULATE_OUTAGE is True


def contrast(foreground, background):
    """WCAG contrast ratio between two #RRGGBB colors."""
    def luminance(color):
        """Relative luminance of a #RRGGBB color."""
        channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    high, low = sorted([luminance(foreground), luminance(background)], reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_light_mode_forced_and_banner_readable():
    """The page forces light mode; the banner text is dark grey on light green or light grey (WCAG AA contrast)."""
    assert '"__theme", "light"' in app.FORCE_LIGHT_JS
    assert f".banner, .banner * {{color: {config.COLOR_TEXT} !important;}}" in app.CSS
    assert contrast(config.COLOR_TEXT, config.COLOR_LIGHT) >= 4.5
    assert contrast(config.COLOR_TEXT, config.COLOR_VISITOR_BG) >= 4.5


def test_chatbot_keeps_consecutive_messages_separate():
    """The chatbot does not merge consecutive assistant messages into one bubble."""
    import gradio as gr
    chatbot = next(b for b in app.build_app().blocks.values() if isinstance(b, gr.Chatbot))
    assert chatbot.group_consecutive_messages is False


def test_welcome_message_on_new_session_and_after_clear():
    """A new session and Clear both show the English welcome message; it never enters the session history."""
    welcome = {"role": "assistant", "content": config.MESSAGES["welcome"]["en"]}
    state = app.new_state()
    assert state["chat"] == [welcome]
    assert state["session"]["history"] == []
    chat, _, cleared, *_ = app.handle_clear(config.MODE_CUSTOMER)
    assert chat == [welcome] and cleared["chat"] == [welcome]
    assert cleared["session"]["history"] == []
    assert welcome not in cleared["session"]["history"]


def test_switching_to_visitor_starts_new_session():
    """Switching to visitor starts a new session: new id, no pending action, hidden card, grey banner, empty panel,
    visitor welcome; the audit records both session ids; switching back does the same (no API call)."""
    state = app.new_state()
    old_id = state["session"]["session_id"]
    state["session"]["pending_action"] = {"action": "lock_card", "card_id": "CARD-1001",
                                          "card_label": "Debit card ****4821", "idempotency_key": "key-1"}
    chat, state, banner, panel, raw, card, card_text, rows = app.handle_mode(config.MODE_VISITOR, state)
    session = state["session"]
    assert session["session_id"] != old_id and session["authenticated"] is False
    assert session["pending_action"] is None and session["history"] == []
    assert card["visible"] is False and card_text == "" and panel == "" and raw is None
    assert 'class="banner visitor"' in banner and config.MODE_VISITOR in banner
    assert chat == state["chat"] == app.welcome_chat("en", authenticated=False)
    assert rows[0][1] == "mode_switch" and old_id in rows[0][4] and session["session_id"] in rows[0][4]
    chat, back, banner, *_ = app.handle_mode(config.MODE_CUSTOMER, state)
    assert back["session"]["session_id"] != session["session_id"] and back["session"]["authenticated"] is True
    assert 'class="banner customer"' in banner and chat == app.welcome_chat("en")


def shows_balance(messages):
    """True if any message shows one of C001's balances, whatever the number format (2,450.75 / 2 450,75)."""
    digits = "".join(c for m in messages for c in m["content"] if c.isdigit() or c == " ")
    return any(b in digits.replace(" ", "") for b in ("245075", "12800"))


def test_no_balance_left_after_switching_to_visitor():
    """After a balance answer, switching to visitor leaves no balance in the chat or in the session history."""
    chat, state, *_ = app.handle_message("What is my balance?", app.new_state())
    assert shows_balance(chat) and shows_balance(state["session"]["history"])  # the test starts from a real balance
    chat, state, *_ = app.handle_mode(config.MODE_VISITOR, state)
    assert not shows_balance(chat) and not shows_balance(state["session"]["history"])
    assert state["session"]["history"] == [] and chat == app.welcome_chat("en", authenticated=False)


def test_new_message_clears_pending_and_hides_card():
    """Any new message other than a confirmation word clears the pending action and hides the Confirm card."""
    state = app.new_state()
    state["session"]["pending_action"] = {"action": "lock_card", "card_id": "CARD-1001",
                                          "card_label": "Debit card ****4821", "idempotency_key": "key-1"}
    chat, state, panel, raw, card, card_text, _ = app.handle_message("merci", state)
    assert state["session"]["pending_action"] is None
    assert card["visible"] is False and card_text == ""
    assert any(step["step"] == "pending_cleared" for step in raw)
    assert bank_api.list_cards(config.CUSTOMER_ID)["cards"][0]["status"] == "active"
