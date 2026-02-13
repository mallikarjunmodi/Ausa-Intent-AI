"""
live_asr.py — Real-time Microphone → Intent Pipeline

Captures audio from the system microphone in short chunks, runs the
full two-tier NLU pipeline (ASR → Domain → Action → Entities → Router).

Usage:
    python3 live_asr.py                # full pipeline (default)
    python3 live_asr.py --chunk 5      # 5-second chunks
    python3 live_asr.py --asr-only     # only ASR, skip NLU + Router

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import io
import time
import logging
import sys
import tempfile
import wave

import numpy as np
import sounddevice as sd

from src.audio.transcriber import WhisperTranscriber
from src.nlu.extractor import IntentExtractor, PipelineResult
from src.router.handler import route

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # keep quiet — only show transcriptions
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000   # 16 kHz — Whisper's native rate
CHANNELS: int = 1           # mono


def record_chunk(duration: float) -> np.ndarray:
    """Record a chunk of audio from the default microphone."""
    print(f"  🎤  Listening for {duration:.0f}s … speak now!", flush=True)
    audio = sd.rec(
        int(SAMPLE_RATE * duration),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )
    sd.wait()
    return audio


def save_to_temp_wav(audio: np.ndarray) -> str:
    """Write a NumPy audio array to a temporary .wav file."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return tmp.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Live microphone pipeline")
    parser.add_argument(
        "--chunk", type=float, default=3.0,
        help="Duration of each recording chunk in seconds (default: 3)",
    )
    parser.add_argument(
        "--asr-only", action="store_true",
        help="Only run ASR (skip NLU + Router)",
    )
    args = parser.parse_args()

    print("\n" + "▓" * 60)
    print("  AUSA HEALTH — Live Microphone Pipeline")
    print("  Press Ctrl+C to stop")
    print("▓" * 60 + "\n")

    # Load models once
    print("  ⏳  Loading Whisper model (tiny.en) …")
    asr = WhisperTranscriber()
    print("  ✅  Whisper ready.\n")

    nlu = None
    if not args.asr_only:
        print("  ⏳  Loading GLiNER model …")
        nlu = IntentExtractor()
        print("  ✅  GLiNER ready.\n")

    chunk_num = 0
    try:
        while True:
            chunk_num += 1
            print(f"\n{'─' * 60}")
            print(f"  Chunk #{chunk_num}")
            print(f"{'─' * 60}")

            # 1. Record from mic
            audio_data = record_chunk(args.chunk)

            # 2. Save to temp .wav
            wav_path = save_to_temp_wav(audio_data)

            # 3. Transcribe
            try:
                t0 = time.perf_counter()
                result = asr.transcribe(wav_path)
                asr_ms = (time.perf_counter() - t0) * 1000

                print(f"  📝  \"{result.text}\"")
                print(f"      (lang={result.language}, dur={result.duration:.1f}s)")
                print(f"  ⏱️  ASR latency: {asr_ms:.0f}ms")

                # 4. Run full NLU + Router pipeline
                nlu_ms = 0.0
                route_ms = 0.0
                if nlu is not None:
                    t1 = time.perf_counter()
                    analysis: PipelineResult = nlu.analyse(result.text)
                    nlu_ms = (time.perf_counter() - t1) * 1000

                    print(f"  🏷️   Domain: {analysis.domain or '(none)'}")
                    print(f"  🎯  Action: {analysis.action or '(none)'}")
                    if analysis.entities:
                        for ent in analysis.entities:
                            print(f"      {ent.label:18s} = {ent.text!r}  ({ent.score:.2f})")
                    if analysis.filled_args:
                        print("  📋  Filled:")
                        for k, v in analysis.filled_args.items():
                            print(f"      ✓ {k} = {v!r}")
                    if analysis.missing_fields:
                        print("  ❗  Missing:")
                        for f in analysis.missing_fields:
                            print(f"      ✗ {f}")
                    print(f"  ⏱️  NLU latency: {nlu_ms:.0f}ms")

                    # Route
                    t2 = time.perf_counter()
                    route(analysis)
                    route_ms = (time.perf_counter() - t2) * 1000
                    print(f"  ⏱️  Route latency: {route_ms:.0f}ms")

                total_ms = asr_ms + nlu_ms + route_ms
                print(f"  ⏱️  TOTAL pipeline: {total_ms:.0f}ms")

            except RuntimeError as exc:
                print(f"  ⚠️  {exc}")

            # Clean up temp file
            import os
            os.unlink(wav_path)

    except KeyboardInterrupt:
        print("\n\n  🛑  Stopped by user. Goodbye!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
