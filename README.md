# Retail banking assistant (prototype)

A web demo of a retail banking customer assistant built as a small multi-agent system on the Mistral API. It answers questions about the bank's services from its documents (with sources), reads the logged-in customer's accounts, gives their branch's opening hours, and locks a card after explicit confirmation. It answers in French or English.

Everything AI is real (Mistral API). Everything bank-side is mocked behind the interfaces a bank would expose. The agent logic stays the same in production: going live replaces the mocks and the in-memory store, and adds strong authentication, an API gateway and monitoring.

## Design principles

1. **The model proposes, the code decides.** The router picks a route and the agents pick tools, but the code checks every decision that matters. Only `lock_card` is allowlisted, and the code picks the card from what the customer named. A visitor cannot reach the account or action agents.
2. **Identity comes from the session.** The customer id stands for the id read server-side from the session token. Tool schemas never contain `customer_id`, and the code injects it when it runs a tool. Text that names another customer is refused before any agent runs.
3. **A confirmation is only valid for what is on screen.** A lock is only proposed until the customer clicks Confirm (simulated strong authentication). Any new message clears the proposal, and typing "yes" never executes anything. The idempotency key ensures one click equals one action.
4. **Never invent, never a dead end.** FAQ answers come only from the retrieved documents and cite them. Every amount comes from a tool result. If the documents do not have the answer, or a service is down, the assistant says so and offers an advisor.
5. **Everything is traced and tested.** Every guardrail result, route, model call, tool call and action goes to the trace panel and to a SQLite audit log, with PII masked. Every bug found in manual testing has a regression test.

## Architecture

```
user message (Gradio UI, app.py)
  │
  ├─ courtesy word, or "yes" typed while an action is pending ──► fixed reply, no router call
  │
  ├─ input guardrails  ║  router            (run in parallel)
  │  moderation +          one JSON call:
  │  injection check       faq | account | action | handoff | out_of_scope | help
  │  flagged ──► fixed refusal, route discarded
  │
  ├─ reply to a cleared card proposal ("not that card") ──► handled in code
  │
  ▼ route (agents.py)
  ├─ faq ──────────► RAG (rag.py: mistral-embed, numpy cosine, threshold) ──► answer + "Source: <title>"
  ├─ account ──────► tool loop ──► mocked bank APIs (read-only, customer id injected by code)
  ├─ action ───────► card check in code ──► proposal only ──► [Confirm button] ──► bank_api.lock_card
  ├─ handoff ──────► 3-line summary for the advisor (trace and audit only)
  └─ help / out_of_scope / visitor on account or action ──► fixed templates
  │
  ▼
output moderation (skipped for fixed templates written by code)
  │
  ▼
audit log (audit.py, SQLite, PII masked) + "How the assistant handled this" panel
```

| File | Role |
|---|---|
| `config.py` | every constant: models, thresholds, customer id, FR/EN message templates, word lists, UI texts |
| `bank_api.py` + `data/bank_data.json` | mocked bank APIs: balances, transactions, branch hours, cards, lock |
| `knowledge/` | 13 short invented FAQ documents |
| `rag.py` | section chunking, embeddings cached in `data/index.npz`, cosine search |
| `guardrails.py` | Mistral moderation, rule-based injection check, PII masking |
| `agents.py` | language detection, router, FAQ, account, action and handoff agents, turn handling, confirm and cancel |
| `audit.py` | SQLite audit log |
| `app.py` | Gradio UI only |
| `eval.py` + `data/golden_set.json` | routing and behaviour evaluation on 32 cases |

## How to run

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env        # then paste your Mistral API key
python app.py               # the demo UI
pytest -v --tb=short        # the tests (they call the real API)
python eval.py              # the golden-set evaluation, writes eval_results.csv
```

## Real vs mocked

| Real | Mocked |
|---|---|
| Mistral chat models for the router and agents (`mistral-small-latest`, temperature 0) | Bank APIs: Python functions with the same shape as a bank's internal services, backed by `data/bank_data.json` |
| Embeddings (`mistral-embed`) and retrieval over the FAQ documents | FAQ content: 13 invented documents, no real bank text |
| Moderation (`mistral-moderation-latest`) on input and model-written output | Login: a fixed demo customer (C001), or visitor mode; no real authentication |
| Function calling, tool loop, confirmation flow, idempotency, audit log | Strong authentication: the Confirm button stands for it |
| Response latencies measured on every step | Vector store: an in-memory numpy index cached on disk |
| | Bank outage: a "Simulate bank API outage" switch in the demo controls |

## How it was tested

- **pytest:** 86 tests (`pytest -v --tb=short`). They run against the real Mistral API, never skip, and fail if the key is missing. They cover the mocked bank, RAG, guardrails, every route, confirm and cancel, visitor mode, the outage path and the UI handlers (called directly, without a browser).
- **Golden set:** `python eval.py` runs the real pipeline on 32 cases in `data/golden_set.json`. They cover every route in English and French, plus `blocked`, `courtesy` and `login_required`, and every regression from manual testing. Each case can have a setup (earlier turns, visitor mode, a locked card, a pending action) and checks such as answer language, "Source:" present, no pending action, or no bank tool call. The script prints routing accuracy, checks passed, and median and p90 latency per route. Network or API errors are counted separately and are not scored as failures.
- **Manual testing in the UI:** every bug found, with its root cause, fix and regression test, is listed in [docs/BUGS.md](docs/BUGS.md).

## Limits

- Session state is in memory: a restart resets the bank data and the conversations.
- The injection check is rule-based. Production would use a dedicated classifier, and would also screen retrieved documents and tool results, which are trusted here.
- The language detection is a simple French/English word heuristic. Anything else gets English answers.
- The golden set is small (32 cases) and was written by the same person as the prompts, so it shows regressions, not real-world accuracy.
- Turns that call an agent model take about 2 to 5 seconds. Answers are not streamed.
- One allowlisted action (temporary card lock). Transfers, unlocking and changes go to the procedure or an advisor.

## Path to production

- Bank APIs behind an API gateway, in place of `bank_api.py`, with the same function contracts.
- A vector store with hybrid search (keyword + embeddings) in place of the numpy index, kept in sync with the document base.
- Mistral models deployed in the bank's environment, so that conversation data stays inside it.
- Real strong authentication on Confirm, and the customer id taken from the session token.
- Monitoring: latency and error dashboards from the audit log, and LLM-as-judge scoring on sampled conversations, plus the golden set run on every change.
