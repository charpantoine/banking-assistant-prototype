"""Gradio UI only: chat, pending-action card, trace panel and audit log. All logic lives in agents.py."""
import gradio as gr

import agents
import audit
import config

AUDIT_COLUMNS = ["ts", "step", "agent", "name", "args_masked", "result_summary", "latency_ms"]
GUARDRAIL_STEPS = {"input_moderation", "injection_check", "output_moderation", "local_reply"}

# Light mode whatever the OS setting: reload once with Gradio's ?__theme=light URL parameter.
FORCE_LIGHT_JS = """() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "light") {
        url.searchParams.set("__theme", "light");
        window.location.replace(url.href);
    }
}"""

CSS = f"""
footer {{display: none !important;}}
.pending-card {{border: 2px solid {config.COLOR_PRIMARY} !important; border-radius: 12px !important;
                background: {config.COLOR_CARD} !important;}}
"""


def make_theme():
    """Soft theme with the green accent, light green background, white cards, dark grey text, rounded corners,
    Inter font from Google Fonts (system sans-serif if it cannot be loaded)."""
    font = (gr.themes.GoogleFont(config.FONT), "ui-sans-serif", "system-ui", "sans-serif")
    return gr.themes.Soft(primary_hue="green", radius_size="lg", font=font).set(
        body_background_fill=config.COLOR_LIGHT, body_text_color=config.COLOR_TEXT,
        block_background_fill=config.COLOR_CARD, color_accent=config.COLOR_PRIMARY,
        button_primary_background_fill=config.COLOR_PRIMARY, button_primary_background_fill_hover=config.COLOR_DARK,
        button_primary_text_color="#FFFFFF", checkbox_background_color_selected=config.COLOR_PRIMARY)


def welcome_chat(lang, authenticated=True):
    """A chat holding only the welcome message for the mode: display-only, never added to the session history."""
    key = "welcome" if authenticated else "welcome_visitor"
    return [{"role": "assistant", "content": config.MESSAGES[key][lang]}]


def new_state(authenticated=True):
    """Fresh per-browser state: a new agent session (new session_id) in the given mode and the welcome message
    in the session's default language (English)."""
    session = agents.new_session(authenticated)
    return {"session": session, "chat": welcome_chat(session["lang"], authenticated)}


def audit_rows():
    """The last 10 audit events as table rows, newest first."""
    return [[event[c] for c in AUDIT_COLUMNS] for event in audit.last_events(config.AUDIT_SHOWN)]


def _line(step):
    """One trace step as a Markdown bullet: agent, name, masked arguments, result and latency."""
    args = "" if step["args"] in ("{}", "") else f" `{step['args'][:120]}`"
    return f"- **{step['agent']}** · {step['name']}{args} → {step['result_summary'][:160]} ({step['latency_ms']:.0f} ms)"


def trace_markdown(trace):
    """Right panel for one turn: route and reason, guardrails, model and tool calls, FAQ sources, total time."""
    route = next((s["result_summary"] for s in trace if s["step"] in ("route", "route_discarded")), "none")
    guards = [_line(s) for s in trace if s["step"] in GUARDRAIL_STEPS]
    calls = [_line(s) for s in trace if s["step"] not in GUARDRAIL_STEPS | {"route", "route_discarded", "answer"}]
    sources = list(dict.fromkeys(t for s in trace if s["step"] == "model_call" for t in s["sources"]))
    total = next((s["latency_ms"] for s in trace if s["step"] == "answer"), 0)
    return "\n".join([f"**Route:** {route}", "", "**Guardrails**", *guards, "", "**Model and tool calls**",
                      *(calls or ["- none"]), "", f"**Sources:** {', '.join(sources) or 'none'}", "",
                      f"**Total time:** {total / 1000:.2f} s"])


def action_markdown(button, answer, session, pending):
    """Right panel after Confirm or Cancel: what the button did, the bank call read from the audit log,
    and, when the bank was called, the idempotency key prefix and who executed the action."""
    events = [e for e in audit.last_events(config.AUDIT_SHOWN) if e["session_id"] == session["session_id"]]
    bank = events[1] if len(events) > 1 and events[1]["step"] == "action" else None
    call = f"lock_card → {bank['result_summary']} ({bank['latency_ms']:.0f} ms)" if bank else "none"
    lines = ["**Action**", f"- **Button:** {button}", "- **Strong authentication:** simulated (button click)",
             f"- **Bank call:** {call}"]
    if bank and pending:
        lines += [f"- **Idempotency key:** `{pending['idempotency_key'][:8]}…`",
                  "- executed by code after customer confirmation"]
    return "\n".join(lines + [f"- **Result:** {answer}"])


def pending_card(pending):
    """Show the Confirm/Cancel card only while an action is pending, with its description."""
    text = (f"**Pending action:** lock your {pending['card_label']} (temporary, reversible lock). "
            "Nothing happens until you confirm.") if pending else ""
    return gr.update(visible=pending is not None), text


def set_busy(busy):
    """Disable (busy) or re-enable the Send and example buttons, in the order [send, *examples]."""
    return [gr.update(interactive=not busy)] * (1 + len(config.EXAMPLE_QUESTIONS))


def show_question(message, state):
    """Step 1 of Send, Enter and example buttons, returns at once: question and "…" in the chat (display only,
    state["chat"] unchanged), input box cleared and disabled, question kept for step 2, buttons disabled."""
    shown = state["chat"]
    if message and message.strip():
        shown = shown + [{"role": "user", "content": message},
                         {"role": "assistant", "content": config.THINKING_PLACEHOLDER}]
    return shown, gr.update(value="", interactive=False), message, *set_busy(True)


def handle_message(message, state):
    """Step 2, the single handler for the agent call: one turn, then chat, panel, card and audit refreshed."""
    if not message or not message.strip():
        return state["chat"], state, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    answer, trace, pending = agents.handle_turn(message, state["session"])
    state["chat"] += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    return state["chat"], state, trace_markdown(trace), trace, *pending_card(pending), audit_rows()


def unlock_inputs():
    """Step 3: re-enable the input box and the buttons; a separate step so it also runs if the agent call failed."""
    return gr.update(interactive=True), *set_busy(False)


def _after_button(button, run, state):
    """Common part of Confirm and Cancel: run the agents function, then message in the chat, action summary
    (raw trace emptied: the buttons do not go through a turn), card state, audit."""
    pending = state["session"]["pending_action"]  # read before run() clears it
    answer = run(state["session"])
    state["chat"].append({"role": "assistant", "content": answer})
    return (state["chat"], state, action_markdown(button, answer, state["session"], pending), None,
            *pending_card(state["session"]["pending_action"]), audit_rows())


def handle_confirm(state):
    """Confirm button: the pending action is executed by code (agents.confirm_action), never by the model."""
    return _after_button(config.CONFIRM_LABEL, agents.confirm_action, state)


def handle_cancel(state):
    """Cancel button: the pending action is cleared, nothing is executed."""
    return _after_button(config.CANCEL_LABEL, agents.cancel_action, state)


def handle_clear(mode):
    """Clear conversation: new session_id in the current mode, welcome message, no pending action, panel hint."""
    state = new_state(mode == config.MODE_CUSTOMER)
    return state["chat"], "", state, config.PANEL_EMPTY, None, *pending_card(None), audit_rows()


def handle_mode(mode, state):
    """Session toggle: any switch starts a new session (audited with both ids) with the new mode's welcome message;
    the trace panel shows its hint and the Confirm card is hidden."""
    authenticated = mode == config.MODE_CUSTOMER
    session = agents.switch_mode(state["session"], authenticated)
    state = {"session": session, "chat": welcome_chat(session["lang"], authenticated)}
    return state["chat"], state, config.PANEL_EMPTY, None, *pending_card(None), audit_rows()


def set_outage(enabled):
    """Demo switch: every mocked bank API answers service_unavailable while on (whole app, not per user)."""
    config.SIMULATE_OUTAGE = bool(enabled)


def demo_label(mode, outage):
    """Demo controls accordion label showing the current state, visible even when closed,
    e.g. "Demo controls · Session: visitor · Outage ON"."""
    parts = [config.DEMO_CONTROLS_TITLE, f"{config.SESSION_LABEL} {config.SESSION_STATES[mode]}"]
    if outage:
        parts.append(config.OUTAGE_ON)
    return " · ".join(parts)


def update_demo_label(mode, outage):
    """Refresh the Demo controls accordion label after a mode or outage change."""
    return gr.update(label=demo_label(mode, outage))


def build_app():
    """Lay out the page and wire every event to the handlers above."""
    with gr.Blocks(title=config.APP_TITLE) as demo:
        state = gr.State(new_state)
        question = gr.State("")  # the question, handed from step 1 to step 2 after the box is cleared
        gr.Markdown(f"# {config.APP_TITLE}\n{config.APP_SUBTITLE}")
        with gr.Row():
            with gr.Column(scale=3):
                # one bubble per message: the Confirm/Cancel result stays separate from the proposal
                chatbot = gr.Chatbot(value=welcome_chat("en"), label="Conversation", height=460,
                             group_consecutive_messages=False)
                with gr.Group(visible=False, elem_classes="pending-card") as card:
                    card_text = gr.Markdown()
                    with gr.Row():
                        confirm = gr.Button(config.CONFIRM_LABEL, variant="primary")
                        cancel = gr.Button(config.CANCEL_LABEL)
                with gr.Row():
                    box = gr.Textbox(placeholder="Ask a question…", show_label=False, scale=5)
                    send = gr.Button("Send", variant="primary", scale=1)
                gr.Markdown(f"**{config.CUSTOMER_QUESTIONS_TITLE}**")
                with gr.Row():
                    examples = [gr.Button(q, size="sm") for q in config.CUSTOMER_EXAMPLES]
                clear = gr.Button(config.CLEAR_LABEL, size="sm")
            with gr.Column(scale=2):
                gr.Markdown(f"### {config.PANEL_TITLE}")
                panel = gr.Markdown(config.PANEL_EMPTY)
                with gr.Accordion(config.RAW_TRACE_TITLE, open=False):
                    raw_trace = gr.JSON()
                with gr.Accordion(config.AUDIT_TITLE, open=False):
                    audit_table = gr.Dataframe(headers=AUDIT_COLUMNS, interactive=False, wrap=True)
                with gr.Accordion(demo_label(config.MODE_CUSTOMER, config.SIMULATE_OUTAGE), open=False) as demo_box:
                    mode = gr.Radio([config.MODE_CUSTOMER, config.MODE_VISITOR], value=config.MODE_CUSTOMER,
                                    label=config.SESSION_LABEL)
                    outage = gr.Checkbox(value=config.SIMULATE_OUTAGE, label=config.OUTAGE_LABEL)
                    examples.append(gr.Button(config.INJECTION_EXAMPLE, size="sm"))
        outputs = [chatbot, state, panel, raw_trace, card, card_text, audit_table]
        busy_buttons = [send, *examples]
        triggers = [(send.click, box), (box.submit, box)] + [(example.click, example) for example in examples]
        for trigger, source in triggers:
            trigger(show_question, [source, state], [chatbot, box, question, *busy_buttons], queue=False).then(
                handle_message, [question, state], outputs, show_progress="hidden").then(
                unlock_inputs, None, [box, *busy_buttons], queue=False)
        confirm.click(handle_confirm, [state], outputs)
        cancel.click(handle_cancel, [state], outputs)
        clear.click(handle_clear, [mode], [chatbot, box, *outputs[1:]])
        mode.change(handle_mode, [mode, state], [chatbot, state, panel, raw_trace, card, card_text, audit_table])
        outage.change(set_outage, [outage], None)
        for control in (mode, outage):
            control.change(update_demo_label, [mode, outage], demo_box, queue=False)
    return demo


if __name__ == "__main__":
    build_app().launch(theme=make_theme(), css=CSS, js=FORCE_LIGHT_JS)
