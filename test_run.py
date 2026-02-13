"""
test_run.py — Multi-Scenario Verification Script

Validates the two-tier NLU pipeline with 8 scenarios covering:
  • Routines  (create, view, delete)
  • Profiles  (update)
  • Appointments (create)
  • Settings  (change)
  • Fallback  (ambiguous input)
  • Missing fields (create routine without time)

Two execution modes:
  • TEXT-ONLY MODE (default) — bypass ASR, feed strings directly
  • AUDIO MODE — generate .wav files, run full pipeline

Run:
    python test_run.py              # text-only (quick validation)
    python test_run.py --audio      # full pipeline with generated .wav
"""

from __future__ import annotations

import logging
import math
import os
import struct
import sys
import wave

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Test Scenarios
# ═══════════════════════════════════════════════════════════════════════════

TEST_SCENARIOS = [
    {
        "name": "Scenario 1 — Create Routine",
        "text": "Create a morning routine to check my blood pressure",
        "expected_domain": "routines",
        "expected_action": "create_routine",
        "expected_entities": "vital=blood pressure, time=morning",
    },
    {
        "name": "Scenario 2 — View Routines",
        "text": "Show me my routines for this week",
        "expected_domain": "routines",
        "expected_action": "view_routines",
        "expected_entities": "timeframe=this week",
    },
    {
        "name": "Scenario 3 — Delete Routine",
        "text": "Delete my blood pressure routine",
        "expected_domain": "routines",
        "expected_action": "delete_routine",
        "expected_entities": "vital=blood pressure",
    },
    {
        "name": "Scenario 4 — Update Profile",
        "text": "Update my profile height to 180 cm",
        "expected_domain": "profiles",
        "expected_action": "update_profile",
        "expected_entities": "height=180 cm",
    },
    {
        "name": "Scenario 5 — Create Appointment",
        "text": "Book an appointment with Dr. Smith on Monday",
        "expected_domain": "appointments",
        "expected_action": "create_appointment",
        "expected_entities": "doctor=Dr. Smith, time=Monday",
    },
    {
        "name": "Scenario 6 — Change Setting",
        "text": "Turn on dark mode",
        "expected_domain": "settings",
        "expected_action": "change_setting",
        "expected_entities": "setting=dark mode",
    },
    {
        "name": "Scenario 7 — Fallback (Ambiguous)",
        "text": "Hello, what can you do?",
        "expected_domain": "(none)",
        "expected_action": "fallback",
        "expected_entities": "—",
    },
    {
        "name": "Scenario 8 — Missing Fields (Create Routine, no time)",
        "text": "Set up a daily reminder to take my medication",
        "expected_domain": "routines",
        "expected_action": "create_routine",
        "expected_entities": "freq=daily, med=medication (should prompt for time)",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Text-Only Mode — NLU → Router (no ASR)
# ═══════════════════════════════════════════════════════════════════════════

def run_text_only_tests() -> None:
    """Feed test strings directly into the NLU → Router pipeline."""
    from src.nlu.extractor import IntentExtractor, PipelineResult
    from src.router.handler import route

    print("\n" + "█" * 60)
    print("  TEXT-ONLY TEST MODE")
    print("  (ASR bypassed — strings fed directly to NLU → Router)")
    print("█" * 60)

    # Initialise NLU model once for all scenarios
    nlu = IntentExtractor()

    for scenario in TEST_SCENARIOS:
        print("\n" + "─" * 60)
        print(f"  🧪  {scenario['name']}")
        print(f"  Input    : \"{scenario['text']}\"")
        print(f"  Expected : domain={scenario['expected_domain']}"
              f"  action={scenario['expected_action']}")
        print(f"  Entities : {scenario['expected_entities']}")
        print("─" * 60)

        # --- Full NLU analysis ---------------------------------------------
        result: PipelineResult = nlu.analyse(scenario["text"])

        print(f"  🏷️   Domain : {result.domain or '(none)'}")
        print(f"  🎯  Action : {result.action or '(none)'}")
        print(f"  🔧  Tool   : {result.tool_name or '(none)'}")
        print("  🔍  Extracted Entities:")
        if result.entities:
            for ent in result.entities:
                print(f"    • {ent.label:18s} = {ent.text!r:30s}  (score={ent.score:.4f})")
        else:
            print("    (none above threshold)")
        print("  📋  Filled Args:")
        if result.filled_args:
            for k, v in result.filled_args.items():
                print(f"    ✓ {k:18s} = {v!r}")
        else:
            print("    (none)")
        if result.missing_fields:
            print("  ❗  Missing Required Fields:")
            for f in result.missing_fields:
                print(f"    ✗ {f}")

        # --- Routing -------------------------------------------------------
        print("  Router output:")
        route(result)

    print("\n" + "█" * 60)
    print("  ALL TEXT-ONLY TESTS COMPLETE")
    print("█" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Audio Mode — Full Pipeline (ASR → NLU → Router)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_sine_wav(filepath: str, duration: float = 2.0, freq: float = 440.0) -> str:
    """Generate a simple sine-wave .wav file (silence placeholder)."""
    sample_rate = 16_000
    n_samples = int(sample_rate * duration)
    amplitude = 16_000

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        for i in range(n_samples):
            sample = int(amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))

    logger.info("Generated placeholder .wav: %s", filepath)
    return os.path.abspath(filepath)


def run_audio_tests() -> None:
    """Generate placeholder .wav files and run the full pipeline."""
    from main import run_pipeline

    print("\n" + "█" * 60)
    print("  AUDIO TEST MODE")
    print("  (Full pipeline: ASR → NLU → Router)")
    print("█" * 60)

    audio_dir = "test_audio"
    wav_files = []

    for i, scenario in enumerate(TEST_SCENARIOS):
        filename = f"{audio_dir}/test_{i + 1}.wav"
        path = _generate_sine_wav(filename, duration=2.0, freq=440.0 + i * 100)
        wav_files.append(path)
        print(f"  Generated: {path}")

    print()
    print("  ⚠️  NOTE: These are sine-wave placeholders, not real speech.")
    print("  For genuine end-to-end tests, replace them with recorded .wav")
    print("  files containing the commands listed in TEST_SCENARIOS.\n")

    for wav_path, scenario in zip(wav_files, TEST_SCENARIOS):
        print("─" * 60)
        print(f"  🧪  {scenario['name']}")
        print(f"  Audio    : {wav_path}")
        print(f"  Expected : domain={scenario['expected_domain']}"
              f"  action={scenario['expected_action']}")
        print("─" * 60)

        try:
            run_pipeline(wav_path)
        except (RuntimeError, Exception) as exc:
            print(f"  ⚠️  Pipeline error (expected for sine-wave): {exc}")

    print("\n" + "█" * 60)
    print("  ALL AUDIO TESTS COMPLETE")
    print("█" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# .wav Generation Instructions
# ═══════════════════════════════════════════════════════════════════════════

def print_wav_instructions() -> None:
    """Print instructions for generating real speech .wav files."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║  HOW TO GENERATE REAL SPEECH .wav FILES FOR TESTING         ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Option A — macOS `say` command (quickest):                  ║
║                                                              ║
║    say -o test_audio/create_routine.wav \\                    ║
║        --data-format=LEI16@16000 \\                           ║
║        "Create a morning routine to check my blood pressure" ║
║                                                              ║
║    say -o test_audio/view_routines.wav \\                     ║
║        --data-format=LEI16@16000 \\                           ║
║        "Show me my routines for this week"                   ║
║                                                              ║
║    say -o test_audio/appointment.wav \\                       ║
║        --data-format=LEI16@16000 \\                           ║
║        "Book an appointment with Dr Smith on Monday"         ║
║                                                              ║
║  Then run:                                                   ║
║    python main.py test_audio/create_routine.wav              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Entry point — choose test mode via CLI flag."""
    if "--audio" in sys.argv:
        run_audio_tests()
    elif "--help-wav" in sys.argv:
        print_wav_instructions()
    else:
        run_text_only_tests()

    # Always print generation instructions at the end
    print_wav_instructions()


if __name__ == "__main__":
    main()
