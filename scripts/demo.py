"""Run both assignment test inputs through the real API and print what the agent did.

Usage:
    python scripts/demo.py               # both examples, in-process (no server needed)
    python scripts/demo.py complex       # just one
    python scripts/demo.py --url http://localhost:8000   # against a running server
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows consoles default to cp1252; LLM text often contains characters such as
# non-breaking hyphens (U+2011) that cp1252 cannot encode.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

ICONS = {"done": "[x]", "failed": "[!]", "skipped": "[-]", "pending": "[ ]", "in_progress": "[~]"}


def show(name: str, body: dict, seconds: float) -> None:
    line = "=" * 78
    print(f"\n{line}\n {name.upper()} REQUEST  ->  {body['status']}  ({seconds:.1f}s)\n{line}")
    plan = body["plan"]
    print(f"Document : {plan['document_type']} - {plan['title']}")
    print(f"Audience : {plan['audience']}")
    if plan["assumptions"]:
        print("\nAssumptions the agent made:")
        for a in plan["assumptions"]:
            print(f"  - {a}")
    if plan["conflicts_resolved"]:
        print("\nConflicts / ambiguities it resolved:")
        for c in plan["conflicts_resolved"]:
            print(f"  - {c}")
    print("\nAgent TODO list (self-generated):")
    for t in body["todo"]:
        detail = f"  -> {t['detail']}" if t["detail"] else ""
        print(f"  {ICONS.get(t['status'], '[?]')} {t['id']:>2}. {t['title']} ({t['agent']}){detail}")
    r = body["review"]
    print(f"\nSelf-review: {r['initial_score']}/10 -> {r['final_score']}/10 "
          f"(threshold {r['threshold']}), revised: {', '.join(r['revised_sections']) or 'none'}")
    for note in r.get("notes", []):
        print(f"  note: {note}")
    llm = body["llm"]
    print(f"\nLLM: {llm['calls']} calls, {llm['retries']} rate-limit/transient retries, "
          f"{llm['json_repairs']} JSON repairs, {llm['total_tokens']:,} tokens")
    print("Work split by agent (successful calls per model):")
    for agent, models in llm["calls_by_agent"].items():
        print(f"  {agent:<9}" + ", ".join(f"{m} x{n}" for m, n in models.items()))
    if llm["tokens_by_model"]:
        print("Tokens per model (load distribution):")
        for model, tokens in llm["tokens_by_model"].items():
            share = 100 * tokens / max(llm["total_tokens"], 1)
            print(f"  {model:<28}{tokens:>8,}  ({share:.0f}%)")
    for w in body["warnings"]:
        print(f"WARNING: {w}")
    print(f"\nSummary : {body['summary']}")
    print(f"Document: {body['document']['download_url']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    available = sorted(p.name.removesuffix("_request.json")
                       for p in (ROOT / "examples").glob("*_request.json"))
    parser.add_argument("which", nargs="*",
                        help=f"examples to run: {', '.join(available)}, or 'all' "
                             "(default: standard complex - the two assignment test inputs)")
    parser.add_argument("--url", help="Base URL of a running server (default: in-process)")
    args = parser.parse_args()
    args.which = available if args.which == ["all"] else (args.which or ["standard", "complex"])
    unknown = set(args.which) - set(available)
    if unknown:
        parser.error(f"unknown example(s): {', '.join(sorted(unknown))}. "
                     f"Choose from: {', '.join(available)}")

    if args.url:
        import httpx
        client = httpx.Client(base_url=args.url, timeout=600)
        context = None
    else:
        from fastapi.testclient import TestClient
        from app.main import app
        context = TestClient(app)
        client = context.__enter__()

    try:
        for name in args.which:
            payload = json.loads((ROOT / "examples" / f"{name}_request.json").read_text("utf-8"))
            started = time.perf_counter()
            response = client.post("/agent", json=payload)
            if response.status_code != 200:
                print(f"{name}: HTTP {response.status_code} {response.text}")
                continue
            show(name, response.json(), time.perf_counter() - started)
        if not args.url:
            print(f"\nFiles saved in: {ROOT / 'outputs'}")
    finally:
        if context is not None:
            context.__exit__(None, None, None)


if __name__ == "__main__":
    main()
