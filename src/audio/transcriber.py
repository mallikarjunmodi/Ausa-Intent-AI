"""
transcriber.py — ASR Node (Faster-Whisper)

Handles local speech-to-text transcription using CTranslate2-backed
Whisper models.  Optimised for edge inference on Qualcomm QCS6490:
  • Model   : tiny.en  (39 M params, English-only)
  • Device  : cpu
  • Compute : int8     (quantised for low-latency edge inference)

Usage:
    from src.audio.transcriber import WhisperTranscriber
    asr = WhisperTranscriber()
    result = asr.transcribe("path/to/audio.wav")
    print(result.text)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract — output of the ASR stage
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TranscriptionResult:
    """Immutable container for a single transcription result.

    Attributes:
        text:     The full transcribed string (stripped & joined segments).
        language: Detected / forced language code (e.g. "en").
        duration: Total audio duration in seconds as reported by the model.
    """
    text: str
    language: str
    duration: float


# ---------------------------------------------------------------------------
# Core ASR class
# ---------------------------------------------------------------------------
class WhisperTranscriber:
    """Edge-optimised speech-to-text engine backed by faster-whisper.

    Parameters:
        model_size: Whisper model variant.  Defaults to ``"tiny.en"`` for
                    minimal memory / latency on the QCS6490.
        device:     Inference device — ``"cpu"`` (default) or ``"cuda"``.
        compute_type: CTranslate2 quantisation level.  ``"int8"`` gives the
                      best trade-off on ARM / edge CPUs.
    """

    # -- Class-level defaults (tuned for QCS6490) ---------------------------
    DEFAULT_MODEL_SIZE: str = "tiny.en"
    DEFAULT_DEVICE: str = "cpu"
    DEFAULT_COMPUTE_TYPE: str = "int8"

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = DEFAULT_DEVICE,
        compute_type: str = DEFAULT_COMPUTE_TYPE,
    ) -> None:
        logger.info(
            "Loading Whisper model  model=%s  device=%s  compute=%s",
            model_size, device, compute_type,
        )
        self._model: WhisperModel = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded successfully.")

    # -- Public API ---------------------------------------------------------

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en",
    ) -> TranscriptionResult:
        """Transcribe a local ``.wav`` file to text.

        Args:
            audio_path: Absolute or relative path to a WAV audio file.
            language:   ISO-639-1 language hint.  Defaults to ``"en"``.

        Returns:
            A ``TranscriptionResult`` dataclass with the transcribed text,
            detected language, and audio duration.

        Raises:
            FileNotFoundError: If *audio_path* does not exist on disk.
            RuntimeError:      If the model returns zero segments (silent /
                               corrupt audio).
        """
        # --- Guard: file must exist ----------------------------------------
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path!r}"
            )

        logger.info("Transcribing audio file: %s", audio_path)

        # --- Run inference -------------------------------------------------
        segments, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=1,           # greedy decoding — fastest on edge
            best_of=1,
            vad_filter=True,       # skip silence for speed
        )

        # Materialise the lazy generator into a single string
        full_text: str = " ".join(
            segment.text.strip() for segment in segments
        ).strip()

        # --- Guard: empty transcription ------------------------------------
        if not full_text:
            logger.warning(
                "Whisper returned an empty transcription for %s", audio_path
            )
            raise RuntimeError(
                f"ASR returned empty transcription for {audio_path!r}.  "
                "The audio may be silent or corrupted."
            )

        logger.info(
            "Transcription complete  lang=%s  duration=%.2fs  chars=%d",
            info.language, info.duration, len(full_text),
        )

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            duration=info.duration,
        )
