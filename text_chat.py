"""
text_chat.py — Interactive Text-to-Intent REPL

Type commands and see the full NLU pipeline output instantly.
No audio/microphone needed.

Usage:
    python3 text_chat.py

Type 'quit' or 'exit' to stop.
"""

from __future__ import annotations

import logging
import sys

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

from src.nlu.extractor import IntentExtractor, PipelineResult
from src.router.handler import route


def main() -> None:
    print("\n" + "▓" * 60)
    print("  AUSA HEALTH — Text-to-Intent REPL")
    print("  Type a command and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("▓" * 60 + "\n")

    print("  ⏳  Loading GLiNER model …")
    nlu = IntentExtractor()
    print("  ✅  Ready!\n")

    while True:
        try:
            text = input("You ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  🛑  Goodbye!\n")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("\n  🛑  Goodbye!\n")
            break

        # Run NLU pipeline
        result: PipelineResult = nlu.analyse(text)

        # Display results
        print(f"\n  🏷️   Domain : {result.domain or '(none)'}")
        print(f"  🎯  Action : {result.action or '(none)'}")
        print(f"  🔧  Tool   : {result.tool_name or '(none)'}")
        if result.entities:
            print("  🔍  Entities:")
            for ent in result.entities:
                print(f"      • {ent.label:18s} = {ent.text!r}  ({ent.score:.2f})")
        if result.filled_args:
            print("  📋  Filled:")
            for k, v in result.filled_args.items():
                print(f"      ✓ {k} = {v!r}")
        if result.missing_fields:
            print("  ❗  Missing:")
            for f in result.missing_fields:
                print(f"      ✗ {f}")

        # Route
        route(result)
        print()


if __name__ == "__main__":
    main()
