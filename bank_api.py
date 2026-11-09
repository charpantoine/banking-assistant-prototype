"""Mocked bank APIs: same shape as a bank's internal services, backed by in-memory data."""
import json
from datetime import datetime
from functools import wraps
from zoneinfo import ZoneInfo

import config

# Loaded once and never written back: a restart resets the demo.
with open(config.BANK_DATA_PATH, encoding="utf-8") as f:
    _data = json.load(f)

# (customer_id, idempotency_key) -> result of the first successful lock.
_locks = {}


def _error(code, message):
    """Build the standard error dict returned by every mocked API."""
    return {"error": code, "message": message}


def _service(name):
    """Decorator giving an API its outage switch, customer check and never-raises guarantee."""
    def decorator(func):
        """Wrap one API function for the service called `name`."""
        @wraps(func)
        def wrapper(customer_id=None, *args, **kwargs):
            """Check the outage flag first, then the customer, then run the function."""
            if config.SIMULATE_OUTAGE:
                return _error("service_unavailable", f"The {name} service is currently unavailable.")
            try:
                if customer_id not in _data["customers"]:
                    return _error("unknown_customer", "This customer does not exist.")
                return func(customer_id, *args, **kwargs)
            except Exception:
                return _error("internal_error", f"The {name} service could not process this request.")
        return wrapper
    return decorator


def _owned(collection, item_id, customer_id):
    """Return the item of this customer with this id, or None (unknown and foreign look the same)."""
    return next((i for i in _data[collection] if i["id"] == item_id and i["customer_id"] == customer_id), None)


@_service("accounts")
def get_balance(customer_id):
    """Return the customer's accounts with label, masked IBAN, balance and currency."""
    accounts = [
        {"id": a["id"], "label": a["label"], "masked_iban": "****" + a["iban"][-4:],
         "balance": a["balance"], "currency": a["currency"]}
        for a in _data["accounts"] if a["customer_id"] == customer_id
    ]
    return {"accounts": accounts}


@_service("transactions")
def get_transactions(customer_id, account_id, limit=config.DEFAULT_TRANSACTIONS):
    """Return the latest transactions of one of the customer's accounts, newest first."""
    account = _owned("accounts", account_id, customer_id)
    if account is None:
        return _error("not_allowed", "This account is not available for this customer.")
    limit = max(1, min(limit, config.MAX_TRANSACTIONS)) if isinstance(limit, int) else config.DEFAULT_TRANSACTIONS
    latest = sorted(account["transactions"], key=lambda t: t["date"], reverse=True)[:limit]
    return {"transactions": [dict(t) for t in latest]}


@_service("branch")
def get_branch_hours(customer_id):
    """Return the customer's own branch with weekly hours and next public holiday closure."""
    branch = _data["branches"][_data["customers"][customer_id]["branch_id"]]
    today = datetime.now(ZoneInfo(config.TIMEZONE)).date().isoformat()
    closure = next((h for h in _data["public_holidays"] if h["date"] >= today), None)
    return {"branch": {**branch, "next_public_holiday_closure": closure}}


@_service("cards")
def list_cards(customer_id):
    """Return the customer's cards with id, type, last 4 digits and status."""
    cards = [
        {"id": c["id"], "type": c["type"], "last_4": c["last_4"], "status": c["status"]}
        for c in _data["cards"] if c["customer_id"] == customer_id
    ]
    return {"cards": cards}


@_service("card locking")
def lock_card(customer_id, card_id, idempotency_key):
    """Lock one of the customer's cards; replaying a successful key returns its first result."""
    key = (customer_id, idempotency_key)
    if key in _locks:
        return dict(_locks[key])
    card = _owned("cards", card_id, customer_id)
    if card is None:
        return _error("not_allowed", "This card is not available for this customer.")
    card["status"] = "locked"
    _locks[key] = {"card_id": card_id, "status": "locked"}
    return dict(_locks[key])
