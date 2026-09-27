"""Measure the bot on a held-out test set (data/eval/test_set.json).

  python -m tools.evaluate                       rule engine, all languages
  python -m tools.evaluate --markdown report.md  also write a report for the project document
  python -m tools.evaluate --with-ai             include the AI fallback (Ollama / API keys)
  python -m tools.evaluate --failures            list every miss

Each question runs in a fresh conversation, so no context carries over and
results don't depend on order. Outcomes for in-scope questions:

  correct   answered with the expected FAQ (or order lookup / small talk)
  clarify   asked "did you mean X?" with the right X - one extra turn, still resolved
  wrong     answered with a different FAQ - the costly failure, counted separately
  missed    escalated to a human though the FAQ had the answer

Off-topic questions must be escalated; answering one is a false answer.
"""

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from dialogue.manager import DialogueManager  # noqa: E402
from nlu.matcher import FAQMatcher  # noqa: E402

TEST_SET = config.DATA_DIR / "eval" / "test_set.json"
LANG_NAMES = {"en": "English", "te": "Telugu", "hi": "Hindi"}


def classify(case: dict, turn) -> str:
    expect = case["expect"]
    if expect == "escalate":
        return "rejected" if turn.action == "escalate" else "false_answer"
    if expect in ("order", "smalltalk"):
        return "correct" if turn.action == expect else "wrong"
    if turn.action in ("answer", "ai") and turn.source == expect:
        return "correct"
    if turn.action == "ai":
        return "correct_ai"  # the LLM answered; judged plausible, not id-matched
    if turn.action == "clarify" and turn.source == expect:
        return "clarify"
    if turn.action in ("answer", "order", "smalltalk", "clarify"):
        return "wrong"
    return "missed"


def run(with_ai: bool = False) -> dict:
    cases = json.load(open(TEST_SET, encoding="utf-8"))["cases"]
    ai = None
    if with_ai:
        from ai.llm import LLMAssistant
        ai = LLMAssistant() or None
    matcher = FAQMatcher()  # shared: building it per question would dominate the timing
    results = []
    for case in cases:
        bot = DialogueManager(matcher=matcher, ai=ai, language=case["lang"])
        start = time.perf_counter()
        turn = bot.handle(case["q"])
        elapsed = (time.perf_counter() - start) * 1000
        results.append({**case, "outcome": classify(case, turn), "action": turn.action,
                        "got": turn.source, "detected_lang": turn.lang, "ms": elapsed})
    return {"results": results, "summary": summarize(results)}


def summarize(results: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in results:
        groups[r["lang"]].append(r)
        groups["all"].append(r)
    summary = {}
    for name, rs in groups.items():
        inscope = [r for r in rs if r["expect"] != "escalate"]
        offtopic = [r for r in rs if r["expect"] == "escalate"]
        n = len(inscope) or 1
        count = lambda outcome: sum(r["outcome"] == outcome for r in inscope)
        correct = count("correct") + count("correct_ai")
        answered = correct + count("wrong")
        summary[name] = {
            "questions": len(rs),
            "in_scope": len(inscope),
            "accuracy": correct / n,
            "resolved_incl_clarify": (correct + count("clarify")) / n,
            "answer_precision": correct / answered if answered else 1.0,
            "wrong": count("wrong"),
            "missed": count("missed"),
            "offtopic": len(offtopic),
            "offtopic_rejected": sum(r["outcome"] == "rejected" for r in offtopic) / (len(offtopic) or 1),
            "language_detected": sum(r["detected_lang"] == r["lang"] for r in rs) / len(rs),
            "latency_ms_avg": statistics.mean(r["ms"] for r in rs),
            "latency_ms_p95": sorted(r["ms"] for r in rs)[int(0.95 * (len(rs) - 1))],
        }
    return summary


ROWS = [("in-scope questions", "in_scope", "{:d}"),
        ("accuracy (answered correctly)", "accuracy", "{:.1%}"),
        ("resolved incl. 'did you mean?'", "resolved_incl_clarify", "{:.1%}"),
        ("answer precision (right when it answers)", "answer_precision", "{:.1%}"),
        ("wrong answers", "wrong", "{:d}"),
        ("missed (sent to human)", "missed", "{:d}"),
        ("off-topic correctly rejected", "offtopic_rejected", "{:.1%}"),
        ("language detected correctly", "language_detected", "{:.1%}"),
        ("avg latency (ms)", "latency_ms_avg", "{:.2f}"),
        ("p95 latency (ms)", "latency_ms_p95", "{:.2f}")]


def table(summary: dict) -> list[str]:
    cols = ["all"] + [l for l in ("en", "te", "hi") if l in summary]
    head = ["metric"] + ["Overall" if c == "all" else LANG_NAMES[c] for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, key, fmt in ROWS:
        lines.append("| " + " | ".join([label] + [fmt.format(summary[c][key]) for c in cols]) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-ai", action="store_true")
    parser.add_argument("--failures", action="store_true")
    parser.add_argument("--markdown", metavar="PATH")
    args = parser.parse_args()
    for stream in (sys.stdout,):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    report = run(with_ai=args.with_ai)
    summary, results = report["summary"], report["results"]
    mode = "rule engine + AI fallback" if args.with_ai else "rule engine only (offline, no AI)"
    lines = [f"# Evaluation: {len(results)} held-out questions, {mode}", "", *table(summary)]
    misses = [r for r in results if r["outcome"] not in ("correct", "correct_ai", "rejected")]
    if misses:
        lines += ["", f"## Not answered correctly ({len(misses)})", "",
                  "| lang | question | expected | outcome | bot did |", "|---|---|---|---|---|"]
        lines += [f"| {r['lang']} | {r['q']} | {r['expect']} | {r['outcome']} | {r['action']} {r['got']} |"
                  for r in misses]
    print("\n".join(lines[:len(table(summary)) + 2] if not args.failures else lines))
    if args.markdown:
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nreport written to {args.markdown}")


if __name__ == "__main__":
    main()
