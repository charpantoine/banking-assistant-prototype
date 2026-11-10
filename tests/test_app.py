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
    assert chat == app.welcome_chat("en") and box == "" and panel == config.PANEL_EMPTY and raw is None
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


def test_light_mode_forced_footer_hidden_and_text_readable():
    """The page forces light mode and hides the Gradio footer; dark grey text on light green or white
    passes WCAG AA contrast."""
    assert '"__theme", "light"' in app.FORCE_LIGHT_JS
    assert "footer {display: none !important;}" in app.CSS
    assert contrast(config.COLOR_TEXT, config.COLOR_LIGHT) >= 4.5
    assert contrast(config.COLOR_TEXT, config.COLOR_CARD) >= 4.5


def test_layout_splits_customer_questions_and_demo_controls():
    """No banner; 5 customer examples (one French) apart from the injection example; Session toggle labelled;
    the trace panel starts with its hint; the theme uses Inter."""
    import gradio as gr
    blocks = list(app.build_app().blocks.values())
    assert not any(isinstance(b, gr.HTML) for b in blocks)
    assert [b.value for b in blocks if isinstance(b, gr.Button) and b.value in config.EXAMPLE_QUESTIONS] == \
        config.CUSTOMER_EXAMPLES + [config.INJECTION_EXAMPLE]
    assert "Quel est le solde de mes comptes ?" in config.CUSTOMER_EXAMPLES
    assert config.INJECTION_EXAMPLE not in config.CUSTOMER_EXAMPLES
    assert next(b for b in blocks if isinstance(b, gr.Radio)).label == config.SESSION_LABEL == "Session:"
    assert any(isinstance(b, gr.Markdown) and b.value == config.PANEL_EMPTY for b in blocks)
    assert app.make_theme().font.startswith("'Inter'")


def in_block(component, block):
    """True if the component is inside the block, at any depth."""
    parent = component.parent
    while parent is not None and parent is not block:
        parent = parent.parent
    return parent is block


def test_demo_controls_accordion_closed_in_right_column_with_state_label(monkeypatch):
    """Demo controls: last accordion of the right column (after Raw trace and Audit log), closed, holding the
    Session toggle, outage checkbox and injection example; customer questions and Clear stay under the chat;
    the label shows session and outage state and is refreshed on change."""
    import gradio as gr
    monkeypatch.setattr(config, "SIMULATE_OUTAGE", False)
    blocks = list(app.build_app().blocks.values())
    demo = next(b for b in blocks if isinstance(b, gr.Accordion) and b.label.startswith(config.DEMO_CONTROLS_TITLE))
    accordions = [c.label for c in demo.parent.children if isinstance(c, gr.Accordion)]
    assert accordions == [config.RAW_TRACE_TITLE, config.AUDIT_TITLE, "Demo controls · Session: logged in"]
    assert demo.open is False
    inside = [b for b in blocks if isinstance(b, (gr.Radio, gr.Checkbox)) or
              (isinstance(b, gr.Button) and b.value == config.INJECTION_EXAMPLE)]
    assert len(inside) == 3 and all(in_block(b, demo) for b in inside)
    outside = [b for b in blocks if isinstance(b, gr.Button) and b.value in config.CUSTOMER_EXAMPLES + [config.CLEAR_LABEL]]
    assert len(outside) == 6 and not any(in_block(b, demo.parent) for b in outside)
    assert app.demo_label(config.MODE_VISITOR, False) == "Demo controls · Session: visitor"
    assert app.demo_label(config.MODE_CUSTOMER, True) == "Demo controls · Session: logged in · Outage ON"
    assert app.update_demo_label(config.MODE_VISITOR, True)["label"] == "Demo controls · Session: visitor · Outage ON"


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
    """Switching to visitor starts a new session: new id, no pending action, hidden card, panel hint,
    visitor welcome; the audit records both session ids; switching back does the same (no API call)."""
    state = app.new_state()
    old_id = state["session"]["session_id"]
    state["session"]["pending_action"] = {"action": "lock_card", "card_id": "CARD-1001",
                                          "card_label": "Debit card ****4821", "idempotency_key": "key-1"}
    chat, state, panel, raw, card, card_text, rows = app.handle_mode(config.MODE_VISITOR, state)
    session = state["session"]
    assert session["session_id"] != old_id and session["authenticated"] is False
    assert session["pending_action"] is None and session["history"] == []
    assert card["visible"] is False and card_text == "" and panel == config.PANEL_EMPTY and raw is None
    assert chat == state["chat"] == app.welcome_chat("en", authenticated=False)
    assert rows[0][1] == "mode_switch" and old_id in rows[0][4] and session["session_id"] in rows[0][4]
    chat, back, panel, *_ = app.handle_mode(config.MODE_CUSTOMER, state)
    assert back["session"]["session_id"] != session["session_id"] and back["session"]["authenticated"] is True
    assert chat == app.welcome_chat("en") and panel == config.PANEL_EMPTY


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
