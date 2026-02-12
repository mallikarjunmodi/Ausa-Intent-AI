"""
main.py — Central Orchestrator

Wires the Audio → Text → Intent → Action pipeline:
    1. WhisperTranscriber  :  .wav  →  raw text
    2. IntentExtractor     :  text  →  AnalysisResult (intent + entities)
    3. Router              :  result →  mock action

Run:
    python main.py path/to/audio.wav
"""

from __future__ import annotations

import logging
import sys

from src.audio.transcriber import WhisperTranscriber, TranscriptionResult
from src.nlu.extractor import IntentExtractor, AnalysisResult
from src.router.handler import route

# ---------------------------------------------------------------------------
# Logging configuration (visible on console for edge debugging)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(audio_path: str) -> None:
    """Execute the full Audio → Text → Intent → Action pipeline.

    Args:
        audio_path: Path to a ``.wav`` file to process.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError:      If ASR produces an empty transcription.
    """
    print("\n" + "▓" * 60)
    print("  AUSA HEALTH — Offline Voice-to-Intent Pipeline")
    print("▓" * 60)

    # ── Stage 1 : ASR ─────────────────────────────────────────────────
    logger.info("━━━  Stage 1 / 3 : Audio → Text  (faster-whisper)  ━━━")
    asr: WhisperTranscriber = WhisperTranscriber()
    transcript: TranscriptionResult = asr.transcribe(audio_path)

    print(f"\n📝  Transcription : \"{transcript.text}\"")
    print(f"    Language      : {transcript.language}")
    print(f"    Duration      : {transcript.duration:.2f}s\n")

    # ── Stage 2 : NLU ─────────────────────────────────────────────────
    logger.info("━━━  Stage 2 / 3 : Text → Intent + Entities  (GLiNER + Keywords)  ━━━")
    nlu: IntentExtractor = IntentExtractor()
    result: AnalysisResult = nlu.analyse(transcript.text)

    print(f"🎯  Intent : {result.intent or '(none)'}")
    print("🔍  Extracted Entities:")
    if result.entities:
        for ent in result.entities:
            print(f"    • {ent.label:18s} = {ent.text!r:30s}  (score={ent.score:.4f})")
    else:
        print("    (none above confidence threshold)")
    print(f"    sensor_target = {result.sensor_target or '(none)'}")
    print(f"    timeframe     = {result.timeframe or '(none)'}")

    # ── Stage 3 : Router ──────────────────────────────────────────────
    logger.info("━━━  Stage 3 / 3 : Result → Action  (Router)  ━━━")
    route(result)

    print("▓" * 60)
    print("  Pipeline complete.")
    print("▓" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI args and launch the pipeline."""
    if len(sys.argv) < 2:
        print("Usage:  python main.py <path_to_wav_file>")
        print("Example:  python main.py test_audio/view_result.wav")
        sys.exit(1)

    audio_path: str = sys.argv[1]

    try:
        run_pipeline(audio_path)
    except FileNotFoundError as exc:
        logger.error("File error: %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 – catch-all for unforeseens
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
