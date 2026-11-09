"""Tests of the mocked bank APIs (no Mistral API call)."""
import bank_api
import config


def card_status(customer_id, card_id):
    """Return the current status of a card as seen through list_cards."""
    return next(c["status"] for c in bank_api.list_cards(customer_id)["cards"] if c["id"] == card_id)


def test_balances_of_c001():
    """C001 sees exactly its two accounts, with balances and masked IBANs only."""
    assert bank_api.get_balance("C001")["accounts"] == [
        {"id": "ACC-1001", "label": "Current account", "masked_iban": "****4567", "balance": 2450.75, "currency": "EUR"},
        {"id": "ACC-1002", "label": "Savings account", "masked_iban": "****6512", "balance": 12800.0, "currency": "EUR"},
    ]


def test_lock_card_locks():
    """lock_card sets the card to locked and leaves the other card active."""
    result = bank_api.lock_card("C001", "CARD-1001", "key-1")
    assert result == {"card_id": "CARD-1001", "status": "locked"}
    assert card_status("C001", "CARD-1001") == "locked"
    assert card_status("C001", "CARD-1002") == "active"


def test_same_idempotency_key_is_one_action():
    """Replaying a key returns the same result and performs no second lock."""
    first = bank_api.lock_card("C001", "CARD-1001", "key-1")
    bank_api._data["cards"][0]["status"] = "active"
    assert bank_api.lock_card("C001", "CARD-1001", "key-1") == first
    assert card_status("C001", "CARD-1001") == "active"


def test_lock_card_of_other_customer_not_allowed():
    """C001 cannot lock C002's card, and the card stays active."""
    assert bank_api.lock_card("C001", "CARD-2001", "key-1")["error"] == "not_allowed"
    assert card_status("C002", "CARD-2001") == "active"


def test_simulate_outage(monkeypatch):
    """With the flag on, every API returns service_unavailable and names its service."""
    monkeypatch.setattr(config, "SIMULATE_OUTAGE", True)
    calls = {
        "accounts": bank_api.get_balance("C001"),
        "transactions": bank_api.get_transactions("C001", "ACC-1001"),
        "branch": bank_api.get_branch_hours("C001"),
        "cards": bank_api.list_cards("C001"),
        "card locking": bank_api.lock_card("C001", "CARD-1001", "key-1"),
    }
    for service, result in calls.items():
        assert result["error"] == "service_unavailable"
        assert service in result["message"]
    monkeypatch.setattr(config, "SIMULATE_OUTAGE", False)
    assert card_status("C001", "CARD-1001") == "active"
