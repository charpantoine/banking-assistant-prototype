"""Every constant of the project. Importing this module also loads .env."""
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
