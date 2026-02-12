"""
extractor.py — NLU Node (GLiNER)

Extracts medical intents and entities from raw text using the
``urchade/gliner_medium-v2.1`` zero-shot NER model.

Key behaviour:
  • Labels are descriptive and tuned for GLiNER's zero-shot NER
    strengths: ``body measurement``, ``time reference``, ``user command``.
  • A 0.4 confidence threshold is used (empirically calibrated — the
    model produces scores in the 0.3–0.9 range for medical text).
  • Intent classification is handled via keyword heuristics on the raw
    text, since GLiNER is an NER model and is weak at extracting
    abstract "intent" labels reliably.

Usage:
    from src.nlu.extractor import IntentExtractor
    nlu = IntentExtractor()
    result = nlu.analyse("Show me my blood pressure results from last week")
    print(result.intent, result.entities)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from gliner import GLiNER

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract — a single extracted entity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractedEntity:
    """Immutable container for one GLiNER prediction.

    Attributes:
        label: The entity label (e.g. ``"body measurement"``, ``"time reference"``).
        text:  The matched surface text from the input string.
        score: Model confidence in ``[0.0, 1.0]``.
    """
    label: str
    text: str
    score: float


# ---------------------------------------------------------------------------
# Core NLU class
# ---------------------------------------------------------------------------
class IntentExtractor:
    """Zero-shot medical intent / entity extractor powered by GLiNER.

    Parameters:
        model_name: HuggingFace model identifier.  Defaults to
                    ``urchade/gliner_medium-v2.1``.
    """

    DEFAULT_MODEL: str = "urchade/gliner_medium-v2.1"

    # Descriptive labels tuned for GLiNER's zero-shot NER strengths.
    # These produce significantly higher confidence scores than abstract
    # labels like "intent" or "sensor_target".
    DEFAULT_LABELS: List[str] = ["body measurement", "time reference", "user command"]

    # Empirically calibrated confidence gate.  GLiNER medium-v2.1 scores
    # for medical text land in the 0.3–0.9 range; 0.4 captures useful
    # entities while filtering noise.
    DEFAULT_THRESHOLD: float = 0.4

    # Keyword sets for heuristic intent classification
    VIEW_KEYWORDS: List[str] = [
        "show", "view", "display", "results", "readings", "check",
        "see", "look", "get", "read", "history", "report",
    ]
    CREATE_KEYWORDS: List[str] = [
        "set up", "create", "schedule", "routine", "start",
        "begin", "new", "add", "plan", "remind", "monitor",
    ]

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        logger.info("Loading GLiNER model: %s", model_name)
        self._model: GLiNER = GLiNER.from_pretrained(model_name)
        logger.info("GLiNER model loaded successfully.")

    # -- Public API ---------------------------------------------------------

    def extract(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[ExtractedEntity]:
        """Extract entities from *text* using zero-shot NER.

        Args:
            text:      The raw input string (e.g. from ASR output).
            labels:    Entity labels to search for.  Falls back to
                       ``DEFAULT_LABELS`` when ``None``.
            threshold: Minimum confidence score.  Predictions below this
                       value are **silently dropped**.

        Returns:
            A list of ``ExtractedEntity`` instances sorted by descending
            confidence score.  May be empty if nothing meets the threshold.
        """
        # --- Resolve labels ------------------------------------------------
        active_labels: List[str] = labels if labels is not None else self.DEFAULT_LABELS

        logger.info(
            "Running GLiNER extraction  labels=%s  threshold=%.2f  text=%r",
            active_labels, threshold, text[:80],
        )

        # --- Run inference -------------------------------------------------
        raw_predictions: List[Dict] = self._model.predict_entities(
            text,
            active_labels,
            threshold=threshold,
        )

        # --- Map to typed dataclass ----------------------------------------
        entities: List[ExtractedEntity] = [
            ExtractedEntity(
                label=pred["label"],
                text=pred["text"],
                score=round(pred["score"], 4),
            )
            for pred in raw_predictions
        ]

        # Sort by confidence (highest first) for deterministic downstream
        entities.sort(key=lambda e: e.score, reverse=True)

        logger.info(
            "GLiNER extracted %d entities above %.2f threshold.",
            len(entities), threshold,
        )
        for ent in entities:
            logger.debug("  → %s = %r  (score=%.4f)", ent.label, ent.text, ent.score)

        return entities

    def classify_intent(self, text: str) -> Optional[str]:
        """Classify user intent via keyword heuristics on the raw text.

        GLiNER is an NER model optimised for entity span extraction, not
        abstract intent classification.  This method supplements GLiNER
        with a simple but reliable keyword scan.

        Args:
            text: Raw input string (from ASR or direct input).

        Returns:
            ``"view_result"``, ``"create_routine"``, or ``None`` if no
            clear intent is detected.
        """
        lower: str = text.lower()

        # Check for create/routine intent FIRST (higher specificity)
        for kw in self.CREATE_KEYWORDS:
            if kw in lower:
                logger.info("Intent classified as 'create_routine' via keyword: %r", kw)
                return "create_routine"

        # Then check for view/result intent
        for kw in self.VIEW_KEYWORDS:
            if kw in lower:
                logger.info("Intent classified as 'view_result' via keyword: %r", kw)
                return "view_result"

        logger.info("No intent keywords matched — returning None.")
        return None

    def analyse(self, text: str) -> "AnalysisResult":
        """Run the full NLU pipeline: entity extraction + intent classification.

        Args:
            text: Raw input string.

        Returns:
            An ``AnalysisResult`` containing the classified intent, extracted
            entities, and convenience accessors for sensor_target / timeframe.
        """
        entities = self.extract(text)
        intent = self.classify_intent(text)

        # Build convenience map: {label: best text}
        entity_map: Dict[str, str] = {}
        for ent in entities:
            if ent.label not in entity_map:
                entity_map[ent.label] = ent.text

        return AnalysisResult(
            intent=intent,
            entities=entities,
            sensor_target=entity_map.get("body measurement"),
            timeframe=entity_map.get("time reference"),
        )


# ---------------------------------------------------------------------------
# Composite analysis result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalysisResult:
    """Full NLU output: intent + extracted entities.

    Attributes:
        intent:        Classified intent string or ``None``.
        entities:      Raw entity list from GLiNER.
        sensor_target: Best-match body measurement text (convenience).
        timeframe:     Best-match time reference text (convenience).
    """
    intent: Optional[str]
    entities: List[ExtractedEntity]
    sensor_target: Optional[str]
    timeframe: Optional[str]
