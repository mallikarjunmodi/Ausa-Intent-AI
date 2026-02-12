"""
test_run.py — Three-Scenario Verification Script

Validates the full Voice-to-Intent pipeline with:
  1. "View result"     → should route to mock_view_result()
  2. "Create routine"  → should route to mock_create_routine()
  3. Ambiguous input   → should trigger mock_fallback_prompt()

Two execution modes:
  • TEXT-ONLY MODE (default) — bypass ASR and feed strings directly
    into the NLU → Router stages.  No .wav files needed.
  • AUDIO MODE — generate .wav files with Python's `wave` module,
    then run the full Audio → Text → Intent → Action pipeline.

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
        "name": "Scenario 1 — View Result",
        "text": "Show me my blood pressure results from last week",
        "expected": "mock_view_result()  (target=blood pressure, timeframe=last week)",
    },
    {
        "name": "Scenario 2 — Create Routine",
        "text": "Set up a morning routine to check my blood pressure",
        "expected": "mock_create_routine()  (target=blood pressure, timeframe=morning)",
    },
    {
        "name": "Scenario 3 — Ambiguous / Fallback",
        "text": "Hello, what can you do?",
        "expected": "mock_fallback_prompt()  (no intent keywords matched)",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Text-Only Mode — NLU → Router (no ASR)
# ═══════════════════════════════════════════════════════════════════════════

def run_text_only_tests() -> None:
    """Feed test strings directly into the NLU → Router pipeline.

    This bypasses the ASR stage entirely, allowing rapid validation
    of intent extraction and routing logic without audio files.
    """
    from src.nlu.extractor import IntentExtractor, AnalysisResult
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
        print(f"  Expected : {scenario['expected']}")
        print("─" * 60)

        # --- Full NLU analysis ---------------------------------------------
        result: AnalysisResult = nlu.analyse(scenario["text"])

        print(f"  Intent: {result.intent or '(none)'}")
        print("  Extracted Entities:")
        if result.entities:
            for ent in result.entities:
                print(f"    • {ent.label:18s} = {ent.text!r:30s}  (score={ent.score:.4f})")
        else:
            print("    (none above threshold)")
        print(f"  sensor_target = {result.sensor_target or '(none)'}")
        print(f"  timeframe     = {result.timeframe or '(none)'}")

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
    """Generate a simple sine-wave .wav file (silence placeholder).

    These files are NOT real speech — they exist so you can verify that
    the ASR stage loads and processes audio without crashing.  For genuine
    end-to-end testing you need real speech recordings.

    Args:
        filepath:  Output .wav path.
        duration:  Length in seconds.
        freq:      Sine frequency in Hz.

    Returns:
        The absolute path to the generated file.
    """
    sample_rate = 16_000  # 16 kHz — Whisper's native rate
    n_samples = int(sample_rate * duration)
    amplitude = 16_000    # ~50 % of int16 max

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)          # mono
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)

        for i in range(n_samples):
            sample = int(amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))

    logger.info("Generated placeholder .wav: %s", filepath)
    return os.path.abspath(filepath)


def run_audio_tests() -> None:
    """Generate placeholder .wav files and run the full pipeline.

    ⚠️  The sine-wave files do NOT contain speech.  Whisper will likely
    produce garbled or empty transcriptions.  This mode is primarily
    useful for verifying the ASR module initialises and runs without
    errors.  For real E2E testing, record actual voice commands.
    """
    from main import run_pipeline

    print("\n" + "█" * 60)
    print("  AUDIO TEST MODE")
    print("  (Full pipeline: ASR → NLU → Router)")
    print("█" * 60)

    audio_dir = "test_audio"
    wav_files = []

    # Generate placeholder .wav files
    for i, scenario in enumerate(TEST_SCENARIOS):
        filename = f"{audio_dir}/test_{i + 1}.wav"
        path = _generate_sine_wav(filename, duration=2.0, freq=440.0 + i * 100)
        wav_files.append(path)
        print(f"  Generated: {path}")

    print()
    print("  ⚠️  NOTE: These are sine-wave placeholders, not real speech.")
    print("  For genuine end-to-end tests, replace them with recorded .wav")
    print("  files containing the commands listed in TEST_SCENARIOS.\n")

    # Run pipeline on each file
    for wav_path, scenario in zip(wav_files, TEST_SCENARIOS):
        print("─" * 60)
        print(f"  🧪  {scenario['name']}")
        print(f"  Audio    : {wav_path}")
        print(f"  Expected : {scenario['expected']}")
        print("─" * 60)

        try:
            run_pipeline(wav_path)
        except (RuntimeError, Exception) as exc:  # noqa: BLE001
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
║    say -o test_audio/view_result.wav \\                       ║
║        --data-format=LEI16@16000 \\                           ║
║        "Show me my blood pressure results from last week"    ║
║                                                              ║
║    say -o test_audio/create_routine.wav \\                    ║
║        --data-format=LEI16@16000 \\                           ║
║        "Set up a morning routine to check my blood pressure" ║
║                                                              ║
║    say -o test_audio/ambiguous.wav \\                         ║
║        --data-format=LEI16@16000 \\                           ║
║        "Hello what can you do"                               ║
║                                                              ║
║  Option B — Python gTTS (requires internet):                 ║
║                                                              ║
║    pip install gTTS pydub                                    ║
║    from gtts import gTTS                                     ║
║    tts = gTTS("Show me my blood pressure results")           ║
║    tts.save("test_audio/view_result.mp3")                    ║
║    # Then convert mp3 → wav with ffmpeg or pydub             ║
║                                                              ║
║  Option C — Record with your microphone:                     ║
║                                                              ║
║    Use any recorder, save as 16 kHz, 16-bit, mono WAV.      ║
║                                                              ║
║  Then run:                                                   ║
║    python main.py test_audio/view_result.wav                 ║
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
