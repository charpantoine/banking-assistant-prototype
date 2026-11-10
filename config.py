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

AGENT_MODEL = "mistral-small-latest"
MAX_ITERATIONS = 5
HISTORY_LIMIT = 20

# Fixed messages, picked by the language of the user's latest message.
MESSAGES = {
    "blocked": {
        "en": "I can't help with that request. I can answer questions about your accounts, your cards and our services.",
        "fr": "Je ne peux pas répondre à cette demande. Je peux vous aider sur vos comptes, vos cartes et nos services.",
    },
    "moderation_unavailable": {
        "en": "The assistant is temporarily unavailable. Please try again in a few minutes.",
        "fr": "L'assistant est temporairement indisponible. Merci de réessayer dans quelques minutes.",
    },
    "output_fallback": {
        "en": "Sorry, I can't give you an answer to that. Please rephrase your question or ask for an advisor.",
        "fr": "Désolé, je ne peux pas vous répondre sur ce point. Merci de reformuler votre question ou de demander un conseiller.",
    },
    "router_error": {
        "en": "Sorry, I didn't understand your request. Please rephrase it or ask for an advisor.",
        "fr": "Désolé, je n'ai pas compris votre demande. Merci de la reformuler ou de demander un conseiller.",
    },
    "out_of_scope": {
        "en": "I can only help with banking questions: your accounts, your cards and our services.",
        "fr": "Je ne peux répondre qu'aux questions bancaires : vos comptes, vos cartes et nos services.",
    },
    "action_not_available": {
        "en": "This action is not available yet in the assistant. An advisor can help you in the meantime.",
        "fr": "Cette action n'est pas encore disponible dans l'assistant. Un conseiller peut vous aider en attendant.",
    },
    "handoff": {
        "en": "An advisor will take over this conversation shortly. I've passed a summary of our conversation "
              "to the advisor, so you won't have to repeat yourself.",
        "fr": "Un conseiller va prendre le relais de cette conversation. Je lui ai transmis un résumé de notre "
              "échange, vous n'aurez donc pas à vous répéter.",
    },
    "faq_not_found": {
        "en": "I couldn't find this information in our documents. An advisor can answer you: just ask to speak to one.",
        "fr": "Je n'ai pas trouvé cette information dans nos documents. Un conseiller peut vous répondre : "
              "il suffit de le demander.",
    },
    "tool_unavailable": {
        "en": "The {service} service is unavailable right now. Please try again in a few minutes or ask for an advisor.",
        "fr": "Le service {service} est indisponible pour le moment. Merci de réessayer dans quelques minutes "
              "ou de demander un conseiller.",
    },
    "tool_error": {
        "en": "The {service} service could not process this request. Please try again or ask for an advisor.",
        "fr": "Le service {service} n'a pas pu traiter cette demande. Merci de réessayer ou de demander un conseiller.",
    },
    "incomplete": {
        "en": "I couldn't complete your request. An advisor can help you.",
        "fr": "Je n'ai pas pu traiter votre demande. Un conseiller peut vous aider.",
    },
}

# Service named in tool error messages, per tool.
SERVICE_NAMES = {
    "get_balance": {"en": "accounts", "fr": "des comptes"},
    "get_transactions": {"en": "transactions", "fr": "des opérations"},
    "get_branch_hours": {"en": "branch", "fr": "des agences"},
    "list_cards": {"en": "cards", "fr": "des cartes"},
    "default": {"en": "bank", "fr": "bancaire"},
}
