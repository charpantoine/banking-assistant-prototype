"""Test of the audit log against a temporary database (no Mistral API call)."""
import audit
import config


def test_log_then_read_newest_first(monkeypatch, tmp_path):
    """Logged events come back with their fields, newest first and limited to n."""
    monkeypatch.setattr(config, "AUDIT_DB_PATH", tmp_path / "audit.db")
    audit.log_event("s1", "tool_call", "account", "get_balance", {"a": 1}, "2 accounts", 12.5)
    audit.log_event("s1", "route", "router", "mistral-small-latest", {"b": 2}, "faq", 30.0)
    events = audit.last_events(1)
    assert len(events) == 1
    assert events[0]["step"] == "route"
    assert events[0]["args_masked"] == '{"b": 2}'
    assert events[0]["result_summary"] == "faq"
    assert events[0]["latency_ms"] == 30.0
    assert len(audit.last_events(10)) == 2


def test_log_event_masks_pii(monkeypatch, tmp_path):
    """An IBAN in the arguments and an email in the summary are stored masked."""
    monkeypatch.setattr(config, "AUDIT_DB_PATH", tmp_path / "audit.db")
    audit.log_event("s1", "input", "user", "message", {"text": "FR7630006000011234567890189"}, "to a@b.fr", 1.0)
    event = audit.last_events(1)[0]
    assert event["args_masked"] == '{"text": "[IBAN]"}'
    assert event["result_summary"] == "to [EMAIL]"
