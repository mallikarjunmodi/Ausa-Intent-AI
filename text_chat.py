#!/usr/bin/env python3
"""
text_chat.py — Interactive text REPL for testing the NLU pipeline.

Type commands and see the agent/tool classification + routing instantly.
When required fields are missing, the system asks for them one by one.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers
for name in ("transformers", "huggingface_hub", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)


def _pretty_field(field_name: str) -> str:
    """Convert 'provider_name' → 'Provider Name'."""
    return field_name.replace("_", " ").title()


def _ask_missing_fields(result) -> bool:
    """Interactively ask the user for each missing required field.

    Returns True if all fields were filled, False if user cancelled.
    """
    print()
    print("─" * 60)
    print(f"  📋  I'm setting up: {result.action}")

    if result.filled_args:
        print("  Already have:")
        for k, v in result.filled_args.items():
            print(f"      ✓ {_pretty_field(k):20s} = {v!r}")

    print("  Still need a few details:")
    print("─" * 60)

    for field in list(result.missing_fields):
        pretty = _pretty_field(field)
        try:
            answer = input(f"  ❓ {pretty} ❯ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  ⏹️  Cancelled.\n")
            return False

        if not answer:
            print(f"      ⚠️  Skipping {pretty} (left empty)")
            continue

        # Fill the slot
        result.filled_args[field] = answer
        result.missing_fields.remove(field)
        print(f"      ✓ {pretty} = {answer!r}")

    print()
    return True


def main() -> None:
    from src.nlu.extractor import IntentExtractor
    from src.router.handler import route, TOOL_DISPATCH

    print()
    print("▓" * 60)
    print("  AUSA HEALTH — Text-to-Intent REPL")
    print("  Type a command and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("▓" * 60)
    print()

    print("  ⏳  Loading GLiNER model …")
    nlu = IntentExtractor()
    nlu._load_model()
    print("  ✅  Ready!\n")

    while True:
        try:
            text = input("You ❯ ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  🛑  Goodbye!\n")
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            print("\n  🛑  Goodbye!\n")
            break

        result = nlu.analyse(text)

        # Display classification
        agent_label = {
            "receptionist": "🏥 Receptionist",
            "nurse": "🩺 Nurse",
            "doctor": "👨‍⚕️ Doctor",
        }
        print()
        print(f"  🤖  Agent  : {agent_label.get(result.agent, '(none)')}")
        print(f"  🎯  Action : {result.action or '(none)'}")
        print(f"  🔧  Tool   : {result.tool_name or '(none)'}")

        if result.entities:
            print("  🔍  Entities:")
            for ent in result.entities:
                print(f"      • {ent.label:22s} = {ent.text!r}  ({ent.score:.2f})")

        if result.filled_args:
            print("  📋  Filled:")
            for k, v in result.filled_args.items():
                print(f"      ✓ {k} = {v!r}")

        # Route to handler
        status = route(result)

        # If missing fields, ask interactively then dispatch
        if status == "missing":
            filled = _ask_missing_fields(result)
            if filled and not result.missing_fields:
                # All fields collected — dispatch the handler
                handler = TOOL_DISPATCH.get(result.action)
                if handler:
                    handler(result.filled_args)
            elif filled:
                # Some fields still empty (user skipped)
                print("  ⚠️  Some fields were skipped — proceeding with partial data.")
                handler = TOOL_DISPATCH.get(result.action)
                if handler:
                    handler(result.filled_args)


if __name__ == "__main__":
    main()
