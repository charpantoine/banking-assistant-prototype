"""Routing evaluation on data/golden_set.json: runs the real pipeline (Mistral API, mocked bank) on each case,
prints a table and a summary, and writes eval_results.csv. Run: python eval.py"""
import csv
import importlib
import json
import math
import statistics
import tempfile
from pathlib import Path

import httpx
from mistralai.client.errors import MistralError, NoResponseError

import agents
import bank_api
import config

GOLDEN_SET = config.BASE_DIR / "data" / "golden_set.json"
RESULTS_CSV = config.BASE_DIR / "eval_results.csv"
API_ERRORS = (httpx.HTTPError, MistralError, NoResponseError)
COLUMNS = ["id", "question", "expected", "got", "route_ok", "checks_ok", "failed_checks", "latency_ms", "error",
           "fixed_template", "output_moderation", "router_ms", "input_moderation_ms", "steps_sum_ms"]


def _class_names(cls):
    """Names of an exception class and all its subclasses (the trace keeps error type names only)."""
    return {cls.__name__}.union(*(_class_names(sub) for sub in cls.__subclasses__()))


API_ERROR_NAMES = set().union(*(_class_names(cls) for cls in API_ERRORS))


def api_error(trace):
    """The network/API failure the pipeline caught in this trace (router, agent or moderation call), else None."""
    for s in trace:
        summary, failure = s["result_summary"], None
        if s["step"] == "error":
            failure = s["name"]
        elif "router failure: " in summary:
            failure = summary.split("router failure: ")[-1].split()[0]
        elif s["step"].endswith("_moderation") and summary.startswith("failed: "):
            failure = summary.removeprefix("failed: ")
        if failure in API_ERROR_NAMES:
            return f"{s['step']}: {failure}"
    return None


def got_route(answer, trace):
    """What the pipeline did: blocked, a local reply (courtesy), login_required, or the router's route.
    A message handled in code after a cleared card proposal ("not that card") counts as action."""
    steps = {s["step"]: s for s in trace}
    if answer in config.MESSAGES["blocked"].values():
        return "blocked"
    if "local_reply" in steps:
        return steps["local_reply"]["name"]
    if "auth_check" in steps:
        return "login_required"
    if "route_discarded" in steps:
        return "action"
    return steps["route"]["result_summary"].split(":")[0]


def check(name, answer, trace, session):
    """True if one expected check holds, e.g. "lang:fr", "contains:Source:", "pending:CARD-1002", "no_bank_call"."""
    kind, _, value = name.partition(":")
    steps = [s["step"] for s in trace]
    tests = {
        "contains": lambda: value in answer,
        "not_contains": lambda: value not in answer,
        "lang": lambda: agents.detect_lang(answer) == value,
        "no_pending": lambda: session["pending_action"] is None,
        "pending": lambda: (session["pending_action"] or {}).get("card_id") == value,
        "tool": lambda: value in [s["name"] for s in trace if s["step"] == "tool_call"],
        "no_bank_call": lambda: "tool_call" not in steps and "card_check" not in steps,
        "no_rag": lambda: "rag_search" not in steps,
        "no_router": lambda: "route" not in steps and "route_discarded" not in steps,
        "handoff_summary": lambda: "handoff_summary" in steps,
        "no_handoff_summary": lambda: "handoff_summary" not in steps,
    }
    return tests[kind]()


def setup_session(setup):
    """Fresh bank data and a new session in the case's mode (logged_in by default), with locked cards if given."""
    importlib.reload(bank_api)  # each case starts from the JSON bank data
    session = agents.new_session(authenticated=setup.get("mode", "logged_in") == "logged_in")
    for card_id in setup.get("locked", []):
        bank_api.lock_card(config.CUSTOMER_ID, card_id, f"eval-{card_id}")
    return session


def set_pending(session, card_id):
    """Put a lock proposal for this card in the session, as propose_lock_card would."""
    card = next(c for c in bank_api.list_cards(config.CUSTOMER_ID)["cards"] if c["id"] == card_id)
    session["pending_action"] = {"action": "lock_card", "card_id": card_id, "card_label": agents._label(card),
                                 "idempotency_key": "eval-key"}


def latency_facts(answer, trace):
    """Per-turn facts for the latency check: fixed template or not, output moderation run or skipped,
    router and input moderation times, and the sum of all step times (to compare with the wall time)."""
    def ms(*names):
        """Latency of the first step with one of these names, 0 if absent."""
        return next((s["latency_ms"] for s in trace if s["step"] in names), 0)
    output = next((s["name"] for s in trace if s["step"] == "output_moderation"), None)
    return {"fixed_template": isinstance(answer, agents.FixedText),
            "output_moderation": "none" if output is None else "skipped" if output == "skipped" else "run",
            "router_ms": ms("route", "route_discarded"), "input_moderation_ms": ms("input_moderation"),
            "steps_sum_ms": round(sum(s["latency_ms"] for s in trace if s["step"] != "answer"))}


def run_case(case):
    """Run the setup (mode, locked cards, earlier turns, pending action), then the question; return a result row.
    A network/API error is recorded in "error" and the case is not scored."""
    setup = case.get("setup", {})
    expected = case["route"] if isinstance(case["route"], list) else [case["route"]]
    row = dict.fromkeys(COLUMNS, "") | {"id": case["id"], "question": case["question"], "expected": "|".join(expected)}
    try:
        session = setup_session(setup)
        for turn in setup.get("turns", []):  # earlier turns run through the real pipeline but are not scored
            _, trace, _ = agents.handle_turn(turn, session)
            if api_error(trace):
                return row | {"error": f"setup {api_error(trace)}"}
        if setup.get("pending"):
            set_pending(session, setup["pending"])
        answer, trace, _ = agents.handle_turn(case["question"], session)
    except API_ERRORS as error:
        return row | {"error": type(error).__name__}
    if api_error(trace):
        return row | {"error": api_error(trace)}
    failed = [c for c in case.get("checks", []) if not check(c, answer, trace, session)]
    got = got_route(answer, trace)
    return row | latency_facts(answer, trace) | {
        "got": got, "route_ok": got in expected, "failed_checks": " ".join(failed),
        "checks_ok": f"{len(case.get('checks', [])) - len(failed)}/{len(case.get('checks', []))}",
        "latency_ms": round(next(s["latency_ms"] for s in trace if s["step"] == "answer"))}


def percentile(values, p):
    """Nearest-rank percentile of a non-empty list (p in 0-100)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def print_row(row):
    """Print one table line: question, expected, got, checks, latency (or the API error)."""
    mark = "ERROR" if row["error"] else ("ok" if row["route_ok"] else "MISS")
    checks = row["error"] or (row["checks_ok"] + (f" FAIL {row['failed_checks']}" if row["failed_checks"] else ""))
    print(f"{row['id']:>3} {row['question'][:44]:<44} {row['expected'][:24]:<24} {row['got']:<14} {mark:<5} "
          f"{checks[:44]:<44} {row['latency_ms']:>7}")


def print_summary(rows):
    """Print routing accuracy, checks passed, API errors, latency per route and the two latency checks."""
    scored, errors = [r for r in rows if not r["error"]], [r for r in rows if r["error"]]
    routed_ok = sum(r["route_ok"] for r in scored)
    passed = sum(int(r["checks_ok"].split("/")[0]) for r in scored)
    total = sum(int(r["checks_ok"].split("/")[1]) for r in scored)
    print(f"\nRouting accuracy: {routed_ok}/{len(scored)} ({100 * routed_ok / max(len(scored), 1):.0f}%)")
    print(f"Checks passed:    {passed}/{total}")
    print(f"Network/API errors (not scored): {len(errors)}" + "".join(f"\n  case {r['id']}: {r['error']}" for r in errors))
    print("\nLatency per route (ms, final turn, from the trace):")
    for name in sorted({r["got"] for r in scored}):
        values = [r["latency_ms"] for r in scored if r["got"] == name]
        print(f"  {name:<14} n={len(values):<3} median={statistics.median(values):>7.0f}  p90={percentile(values, 90):>7.0f}")
    fixed = [r for r in scored if r["fixed_template"]]
    model = [r for r in scored if not r["fixed_template"]]
    print("\nLatency checks:")
    print(f"  Output moderation skipped on fixed templates: {sum(r['output_moderation'] != 'run' for r in fixed)}"
          f"/{len(fixed)}; run on model-written answers: {sum(r['output_moderation'] == 'run' for r in model)}/{len(model)}")
    both = [r for r in scored if r["router_ms"] and r["input_moderation_ms"]]
    if both:
        overlap = statistics.median(r["steps_sum_ms"] - r["latency_ms"] for r in both)
        shorter = statistics.median(min(r["router_ms"], r["input_moderation_ms"]) for r in both)
        verdict = "parallel" if overlap > shorter / 2 else "NOT parallel"
        print(f"  Input guardrails vs router on {len(both)} turns: sum of step times minus wall time = {overlap:.0f} ms "
              f"(median), shorter of the two calls = {shorter:.0f} ms (median) -> {verdict}")


def main():
    """Run every golden case with the audit log in a temporary database, print the table and summary, write the CSV."""
    cases = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    config.AUDIT_DB_PATH = Path(tempfile.mkdtemp()) / "audit.db"  # keep the demo audit log clean
    print(f"{'id':>3} {'question':<44} {'expected':<24} {'got':<14} {'route':<5} {'checks':<44} {'ms':>7}")
    rows = []
    for case in cases:
        rows.append(run_case(case))
        print_row(rows[-1])
    print_summary(rows)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {RESULTS_CSV.name}")


if __name__ == "__main__":
    main()
