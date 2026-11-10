"""Every constant of the project. Importing this module also loads .env and provides the Mistral client."""
import os
from pathlib import Path

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()


def mistral_client():
    """Create a Mistral client with the key from .env; fail with a clear message if it is missing."""
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY missing")
    return Mistral(api_key=key)


BASE_DIR = Path(__file__).parent
BANK_DATA_PATH = BASE_DIR / "data" / "bank_data.json"
AUDIT_DB_PATH = BASE_DIR / "audit.db"

# Stands for the id read server-side from the session token; never taken from the model or the user.
CUSTOMER_ID = "C001"

# Runtime flag: when True, every mocked bank API answers service_unavailable.
SIMULATE_OUTAGE = False

TIMEZONE = "Europe/Paris"

DEFAULT_TRANSACTIONS = 5
MAX_TRANSACTIONS = 50

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
INDEX_PATH = BASE_DIR / "data" / "index.npz"
EMBED_MODEL = "mistral-embed"
EMBED_BATCH_SIZE = 32
TOP_K = 3
# Calibrated on 10 queries; weakest on-topic 0.733 vs best uncovered banking 0.720;
# loose filter by design, precision comes from the grounded FAQ prompt (see test in stage 3).
RAG_THRESHOLD = 0.70

TEMPERATURE = 0
ROUTER_MODEL = "mistral-small-latest"
ROUTES = ("faq", "account", "action", "handoff", "out_of_scope")
ROUTER_HISTORY = 6
MODERATION_MODEL = "mistral-moderation-latest"
# Only these categories block; the others (financial, pii, law, health...) are reported but never block.
MODERATION_BLOCKING = {"sexual", "hate_and_discrimination", "violence_and_threats",
                       "dangerous_and_criminal_content", "selfharm"}
