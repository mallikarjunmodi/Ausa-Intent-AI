"""
extractor.py — Two-Tier NLU Pipeline (GLiNER)

Implements a hierarchical intent classification + entity extraction system:

  Tier 1  →  Domain classification (routines / profiles / appointments / settings)
  Tier 2  →  Sub-action classification per domain (create / view / update / delete)
  Entity  →  Slot filling with domain-specific entity labels
  Slots   →  Map entities into Pydantic model fields + detect missing required fields

Usage:
    from src.nlu.extractor import IntentExtractor
    nlu = IntentExtractor()
    result = nlu.analyse("Create a morning routine to check my blood pressure")
    print(result.domain, result.action, result.filled_args, result.missing_fields)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gliner import GLiNER

from src.models.schemas import TOOL_REGISTRY

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExtractedEntity:
    """One GLiNER prediction."""
    label: str
    text: str
    score: float


@dataclass
class PipelineResult:
    """Full NLU output after two-tier classification + slot filling.

    Attributes:
        domain:         Top-level intent domain (e.g. "routines").
        action:         Sub-action (e.g. "create_routine").
        tool_name:      Function/tool to dispatch to (same as action).
        filled_args:    Dict of successfully filled Pydantic model fields.
        missing_fields: List of required fields still missing a value.
        entities:       Raw extracted entities from GLiNER.
        confidence:     Tier-1 classification confidence.
        raw_text:       The original input text.
    """
    domain: Optional[str] = None
    action: Optional[str] = None
    tool_name: Optional[str] = None
    filled_args: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    entities: List[ExtractedEntity] = field(default_factory=list)
    confidence: float = 0.0
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Intent / Sub-action configuration
# ---------------------------------------------------------------------------

# Tier 1: Top-level domain labels for GLiNER
DOMAIN_LABELS: List[str] = ["routines", "profiles", "appointments", "settings"]
DOMAIN_THRESHOLD: float = 0.2  # Low threshold — short queries score low

# Tier 2: Per-domain sub-action labels and their mapped tool names
SUB_ACTION_CONFIG: Dict[str, Dict] = {
    "routines": {
        "labels": ["create routine", "view routine", "view result", "update routine", "delete routine"],
        "intent_map": {
            "create routine": "create_routine",
            "view routine": "view_routines",
            "view result": "view_result",
            "update routine": "update_routine",
            "delete routine": "delete_routine",
        },
        "default_action": "view_routines",
    },
    "profiles": {
        "labels": ["view profile", "update profile"],
        "intent_map": {
            "view profile": "view_profile",
            "update profile": "update_profile",
        },
        "default_action": "view_profile",
    },
    "appointments": {
        "labels": ["create appointment", "view appointment", "cancel appointment"],
        "intent_map": {
            "create appointment": "create_appointment",
            "view appointment": "view_appointments",
            "cancel appointment": "cancel_appointment",
        },
        "default_action": "view_appointments",
    },
    "settings": {
        "labels": ["change setting", "view setting"],
        "intent_map": {
            "change setting": "change_setting",
            "view setting": "view_settings",
        },
        "default_action": "view_settings",
    },
}

# Entity labels for slot filling (used in the entity extraction pass)
ENTITY_LABELS: List[str] = [
    "vital sign type",
    "time reference",
    "frequency",
    "doctor name",
    "body metric",
    "person name",
    "setting name",
    "setting value",
    "location",
    "medication name",
    "symptom or complaint",
]
ENTITY_THRESHOLD: float = 0.35

# Mapping from entity labels → Pydantic field names (per tool)
ENTITY_TO_FIELD_MAP: Dict[str, Dict[str, str]] = {
    # Routines
    "create_routine": {
        "vital sign type": "vital_test_type",
        "time reference": "scheduled_time",
        "frequency": "frequency",
        "medication name": "routine_name",
    },
    "view_routines": {
        "vital sign type": "category",
        "time reference": "timeframe",
        "frequency": "frequency",
    },
    "view_result": {
        "vital sign type": "vital_type",
        "time reference": "timeframe",
        "body metric": "vital_type",
    },
    "update_routine": {
        "vital sign type": "vital_test_type",
        "time reference": "scheduled_time",
        "frequency": "frequency",
    },
    "delete_routine": {
        "vital sign type": "routine_name",
    },
    # Profiles
    "view_profile": {
        "body metric": "section",
    },
    "update_profile": {
        "person name": "name",
        "body metric": "height",  # simplification; refined below
    },
    # Appointments
    "create_appointment": {
        "doctor name": "doctor_name",
        "time reference": "date_time",
        "location": "location",
        "symptom or complaint": "symptoms",
        "person name": "patient_name",
    },
    "view_appointments": {
        "time reference": "timeframe",
        "doctor name": "doctor_name",
    },
    "cancel_appointment": {
        "doctor name": "doctor_name",
        "time reference": "date_time",
    },
    # Settings
    "change_setting": {
        "setting name": "setting_name",
        "setting value": "setting_value",
    },
    "view_settings": {
        "setting name": "setting_name",
    },
}

# Keyword fallbacks — used when GLiNER confidence is too low
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "routines": [
        "routine", "routines", "schedule", "reminder", "medication",
        "blood pressure", "heart rate", "spo2", "ecg", "temperature",
        "glucose", "vital", "monitor", "check", "measure", "reading",
        "daily", "weekly", "morning", "evening",
    ],
    "profiles": [
        "profile", "my info", "my name", "height", "weight",
        "allergy", "allergies", "diagnosis", "care team", "family",
    ],
    "appointments": [
        "appointment", "appointments", "doctor", "dr.", "book",
        "clinic", "visit", "consultation", "checkup",
        "sick", "feeling", "ill", "unwell", "symptom", "symptoms",
        "pain", "ache", "fever", "cough", "injury",
    ],
    "settings": [
        "setting", "settings", "dark mode", "brightness", "notification",
        "notifications", "text size", "font", "display", "volume",
        "privacy", "connected device",
    ],
}

ACTION_KEYWORDS: Dict[str, List[str]] = {
    "create": ["create", "set up", "add", "new", "start", "begin", "schedule", "book", "make", "remind", "record", "want", "log", "track", "measure"],
    "view": ["show", "view", "display", "see", "look", "get", "list"],
    "result": ["reading", "readings", "result", "results", "last", "past", "previous", "history", "was"],
    "update": ["update", "change", "edit", "modify", "set", "turn", "switch", "enable", "disable"],
    "delete": ["delete", "remove", "cancel", "stop", "end", "clear"],
}


# ---------------------------------------------------------------------------
# Core NLU class
# ---------------------------------------------------------------------------
class IntentExtractor:
    """Two-tier GLiNER intent classifier with entity slot filling."""

    DEFAULT_MODEL: str = "urchade/gliner_medium-v2.1"

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        logger.info("Loading GLiNER model: %s", model_name)
        self._model: GLiNER = GLiNER.from_pretrained(model_name)
        logger.info("GLiNER model loaded successfully.")

    # -- Internal helpers ---------------------------------------------------

    def _gliner_predict(
        self, text: str, labels: List[str], threshold: float
    ) -> List[Dict]:
        """Run GLiNER prediction and return raw results."""
        return self._model.predict_entities(text, labels, threshold=threshold)

    def _classify_domain_gliner(self, text: str) -> tuple[Optional[str], float]:
        """Tier-1: Classify top-level domain using GLiNER."""
        preds = self._gliner_predict(text, DOMAIN_LABELS, DOMAIN_THRESHOLD)
        if not preds:
            return None, 0.0
        # Pick highest-confidence prediction
        best = max(preds, key=lambda p: p["score"])
        return best["text"].lower().strip(), best["score"]

    def _classify_domain_keywords(self, text: str) -> Optional[str]:
        """Fallback Tier-1: Classify domain via keyword matching."""
        lower = text.lower()
        scores: Dict[str, int] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                scores[domain] = score
        if not scores:
            return None
        return max(scores, key=scores.get)

    def _classify_action_hybrid(
        self, text: str, domain: str
    ) -> tuple[Optional[str], float]:
        """Tier-2: Hybrid sub-action classification.

        Combines GLiNER predictions with keyword scoring to avoid
        GLiNER overriding strong keyword signals (e.g. 'record' → create).
        """
        config = SUB_ACTION_CONFIG.get(domain)
        if not config:
            return None, 0.0

        lower = text.lower()

        # ── Keyword scoring ───────────────────────────────────────
        kw_scores: Dict[str, int] = {}
        for action_type, keywords in ACTION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            if score > 0:
                kw_scores[action_type] = score

        best_kw_type = max(kw_scores, key=kw_scores.get) if kw_scores else None
        best_kw_score = kw_scores.get(best_kw_type, 0) if best_kw_type else 0

        # ── GLiNER scoring ────────────────────────────────────────
        preds = self._gliner_predict(text, config["labels"], 0.15)
        gliner_action = None
        gliner_conf = 0.0
        if preds:
            best = max(preds, key=lambda p: p["score"])
            gliner_action = config["intent_map"].get(best["label"])
            gliner_conf = best["score"]

        logger.info(
            "Tier-2 hybrid: gliner=%r(%.3f)  keywords=%r(%d)",
            gliner_action, gliner_conf, best_kw_type, best_kw_score,
        )

        # ── Decision logic ────────────────────────────────────────
        # If keywords have a strong signal (2+ matches) and disagree
        # with GLiNER, trust keywords — GLiNER is unreliable for
        # action verbs on short phrases.
        kw_action = self._map_keyword_type_to_action(best_kw_type, config)

        if best_kw_score >= 2 and kw_action != gliner_action:
            logger.info(
                "Keywords override GLiNER: %r (kw_score=%d) beats %r (conf=%.3f)",
                kw_action, best_kw_score, gliner_action, gliner_conf,
            )
            return kw_action, gliner_conf

        # If GLiNER is confident, use it
        if gliner_action and gliner_conf >= 0.3:
            return gliner_action, gliner_conf

        # Otherwise fall back to keywords
        if kw_action:
            return kw_action, gliner_conf

        return config.get("default_action"), 0.0

    @staticmethod
    def _map_keyword_type_to_action(
        action_type: Optional[str], config: Dict
    ) -> Optional[str]:
        """Map a keyword action type (create/view/result/update/delete)
        to the domain-specific tool name."""
        if action_type is None:
            return None
        action_type_to_label = {
            "create": next((l for l in config["labels"] if "create" in l), None),
            "view": next((l for l in config["labels"] if l.startswith("view") and "result" not in l), None),
            "result": next((l for l in config["labels"] if "result" in l), None),
            "update": next((l for l in config["labels"] if "update" in l or "change" in l), None),
            "delete": next((l for l in config["labels"] if "delete" in l or "cancel" in l), None),
        }
        label = action_type_to_label.get(action_type)
        if label:
            return config["intent_map"].get(label)
        # For domains without a 'result' sub-action, fall back to 'view'
        if action_type == "result":
            view_label = action_type_to_label.get("view")
            if view_label:
                return config["intent_map"].get(view_label)
        return None

    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract named entities for slot filling."""
        preds = self._gliner_predict(text, ENTITY_LABELS, ENTITY_THRESHOLD)
        entities = [
            ExtractedEntity(
                label=p["label"],
                text=p["text"],
                score=round(p["score"], 4),
            )
            for p in preds
        ]
        entities.sort(key=lambda e: e.score, reverse=True)
        return entities

    @staticmethod
    def _extract_symptoms_heuristic(text: str) -> Optional[str]:
        """Extract symptoms/complaints from natural language using patterns.

        Catches phrases GLiNER misses because they're conversational:
          - "I don't like to smile"
          - "I can't sleep at night"
          - "I have pain in my back"
          - "suffering from headaches"
          - "my knee hurts"
        """
        import re
        lower = text.lower()

        # Patterns that signal a symptom/complaint clause
        symptom_patterns = [
            # "I don't/can't/couldn't ..."
            r"i\s+(?:don'?t|can'?t|couldn'?t|cannot|am not able to)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
            # "I'm/Im feeling/having ..."
            r"i(?:'?m| am)\s+(?:feeling|having|experiencing)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
            # "I have/feel/experience ..."
            r"i\s+(?:have|feel|experience|notice|get|got)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
            # "suffering from / troubled by ..."
            r"(?:suffering|troubled|bothered)\s+(?:from|by|with)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
            # "my ... hurts/aches/is swollen"
            r"my\s+(.+?\s+(?:hurts?|aches?|is\s+(?:swollen|sore|painful|stiff|numb)))",
            # "pain in / problem with ..."
            r"(?:pain|ache|problem|issue|trouble|difficulty)\s+(?:in|with)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
        ]

        for pattern in symptom_patterns:
            match = re.search(pattern, lower)
            if match:
                symptom = match.group(1).strip().rstrip(",. ")
                if len(symptom) > 2:  # skip trivially short matches
                    logger.info("Heuristic symptom extracted: %r", symptom)
                    return symptom

        return None

    def _fill_slots(
        self, action: str, entities: List[ExtractedEntity], text: str
    ) -> tuple[Dict[str, Any], List[str]]:
        """Map extracted entities → Pydantic model fields.

        Returns:
            (filled_args, missing_fields)
        """
        field_map = ENTITY_TO_FIELD_MAP.get(action, {})
        model_cls = TOOL_REGISTRY.get(action)

        # Build filled_args from entity → field mapping
        filled: Dict[str, Any] = {}
        for ent in entities:
            field_name = field_map.get(ent.label)
            if field_name and field_name not in filled:
                filled[field_name] = ent.text

        # Heuristic symptom fallback for appointment actions
        if action in ("create_appointment", "view_appointments") and "symptoms" not in filled:
            symptom = self._extract_symptoms_heuristic(text)
            if symptom:
                filled["symptoms"] = symptom

        # Check for missing required fields
        missing: List[str] = []
        if model_cls:
            try:
                instance = model_cls(**filled)
                missing = instance.get_missing_required()
            except Exception:
                # If model validation fails, just report all required as missing
                missing = getattr(model_cls, "REQUIRED_FIELDS", [])

        return filled, missing

    # -- Public API ---------------------------------------------------------

    def analyse(self, text: str) -> PipelineResult:
        """Run the full two-tier NLU pipeline.

        Steps:
            1. Tier-1: Classify domain (GLiNER → keyword fallback)
            2. Tier-2: Classify sub-action (GLiNER → keyword fallback)
            3. Extract entities for slot filling
            4. Fill Pydantic model slots + detect missing required fields

        Args:
            text: Raw input string (from ASR or direct input).

        Returns:
            A ``PipelineResult`` with domain, action, filled_args,
            missing_fields, and raw entities.
        """
        result = PipelineResult(raw_text=text)

        # ── Tier 1: Domain classification ─────────────────────────────
        domain, domain_conf = self._classify_domain_gliner(text)
        logger.info("Tier-1 GLiNER: domain=%r  confidence=%.3f", domain, domain_conf)

        # Validate against known domains
        if domain and domain not in SUB_ACTION_CONFIG:
            # GLiNER returned text that doesn't match a domain label —
            # try fuzzy matching
            for known_domain in DOMAIN_LABELS:
                if domain in known_domain or known_domain in domain:
                    domain = known_domain
                    break
            else:
                logger.info("GLiNER domain %r not in known domains, falling back to keywords.", domain)
                domain = None

        if domain is None or domain_conf < DOMAIN_THRESHOLD:
            domain = self._classify_domain_keywords(text)
            logger.info("Tier-1 keyword fallback: domain=%r", domain)

        if domain is None:
            logger.info("No domain classified — returning fallback result.")
            return result

        result.domain = domain
        result.confidence = domain_conf

        # ── Tier 2: Sub-action classification (hybrid) ─────────────────
        action, action_conf = self._classify_action_hybrid(text, domain)
        logger.info("Tier-2 result: action=%r  confidence=%.3f", action, action_conf)

        if action is None:
            action = SUB_ACTION_CONFIG[domain].get("default_action")
            logger.info("Using default action for domain %r: %r", domain, action)

        result.action = action
        result.tool_name = action

        # ── Entity extraction + slot filling ──────────────────────────
        entities = self._extract_entities(text)
        result.entities = entities

        logger.info("Extracted %d entities:", len(entities))
        for ent in entities:
            logger.info("  → %s = %r  (%.4f)", ent.label, ent.text, ent.score)

        if action:
            filled_args, missing_fields = self._fill_slots(action, entities, text)
            result.filled_args = filled_args
            result.missing_fields = missing_fields

            logger.info("Filled args: %s", filled_args)
            if missing_fields:
                logger.info("Missing required fields: %s", missing_fields)

        return result
