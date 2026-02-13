#!/usr/bin/env python3
"""
main.py — Entry point for the AUSA Voice-to-Intent pipeline.

Supports two modes:
  1. Text-only    — Feed a string directly to NLU (skip ASR)
  2. Audio file   — Transcribe a .wav file, then run NLU

Usage:
  python3 main.py --text "take my blood pressure"
  python3 main.py --audio recording.wav
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

for name in ("transformers", "huggingface_hub", "urllib3"):
    logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    parser = argparse.ArgumentParser(description="AUSA Voice-to-Intent Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Direct text input (skip ASR)")
    group.add_argument("--audio", type=str, help="Path to .wav file")
    args = parser.parse_args()

    from src.nlu.extractor import IntentExtractor
    from src.router.handler import route

    nlu = IntentExtractor()

    if args.audio:
        from src.audio.transcriber import WhisperTranscriber
        transcriber = WhisperTranscriber()
        logger.info("Transcribing: %s", args.audio)
        transcription = transcriber.transcribe(args.audio)
        text = transcription.text
        logger.info("Transcription: %r", text)
        if not text or not text.strip():
            logger.warning("Empty transcription — nothing to classify.")
            sys.exit(1)
    else:
        text = args.text

    print(f"\n  📝  Input: \"{text}\"\n")

    result = nlu.analyse(text)

    agent_label = {
        "receptionist": "🏥 Receptionist",
        "nurse": "🩺 Nurse",
        "doctor": "👨‍⚕️ Doctor",
    }

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

    if result.missing_fields:
        print("  ❗  Missing:")
        for f in result.missing_fields:
            print(f"      ✗ {f}")

    print()
    route(result)


if __name__ == "__main__":
    main()
