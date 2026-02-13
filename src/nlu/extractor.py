"""
extractor.py — Hierarchical NLU pipeline using GLiNER

Two-tier classification:
  Tier 1 → Classify which Agent  (Receptionist / Nurse / Doctor)
  Tier 2 → Classify which Tool   (e.g. appointment.create, takeTest, routine.read)

Hybrid scoring combines GLiNER predictions with keyword matching
so that strong keyword signals override weak GLiNER predictions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.models.schemas import TOOL_REGISTRY

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedEntity:
    label: str
    text: str
    score: float


@dataclass
class PipelineResult:
    """Complete output of the NLU pipeline."""
    raw_text: str
    agent: Optional[str] = None          # receptionist / nurse / doctor
    action: Optional[str] = None         # e.g. appointment.create, takeTest
    tool_name: Optional[str] = None      # same as action (for dispatch)
    entities: List[ExtractedEntity] = field(default_factory=list)
    filled_args: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL: str = "urchade/gliner_medium-v2.1"
DOMAIN_THRESHOLD: float = 0.25
ENTITY_THRESHOLD: float = 0.35

# ── Tier 1: Agent labels ─────────────────────────────────────────────────

AGENT_LABELS: List[str] = [
    "health management",
    "health recording",
    "health consultation",
]

AGENT_MAP: Dict[str, str] = {
    "health management": "receptionist",
    "health recording": "nurse",
    "health consultation": "doctor",
}

# ── Tier 2: Per-agent tool labels and mapped tool names ───────────────────

SUB_ACTION_CONFIG: Dict[str, Dict] = {
    "receptionist": {
        "labels": [
            "read profile", "update profile", "open camera",
            "verify phone", "verify email",
            "read diagnosis", "create allergy", "read allergies",
            "update allergy", "delete allergy",
            "read care team",
            "create family member", "read family", "update family member",
            "delete family member", "view permissions",
            "read wifi", "update brightness", "update text size",
            "connect device", "read devices", "delete device",
            "read notifications", "update notification",
            "update smart prompt",
            "read call settings", "update call settings",
            "create appointment", "read appointments",
            "update appointment", "delete appointment",
        ],
        "intent_map": {
            "read profile": "profile.read",
            "update profile": "profile.update",
            "open camera": "camera.open",
            "verify phone": "profile.verifyPhone",
            "verify email": "profile.verifyEmail",
            "read diagnosis": "diagnosis.read",
            "create allergy": "allergies.create",
            "read allergies": "allergies.read",
            "update allergy": "allergies.update",
            "delete allergy": "allergies.delete",
            "read care team": "careTeam.read",
            "create family member": "family.create",
            "read family": "family.read",
            "update family member": "family.update",
            "delete family member": "family.delete",
            "view permissions": "family.permissionsSchema",
            "read wifi": "wifi.read",
            "update brightness": "brightness.update",
            "update text size": "textSize.update",
            "connect device": "device.connect",
            "read devices": "device.read",
            "delete device": "device.delete",
            "read notifications": "notification.read",
            "update notification": "notification.update",
            "update smart prompt": "smartPrompt.update",
            "read call settings": "callSettings.read",
            "update call settings": "callSettings.update",
            "create appointment": "appointment.create",
            "read appointments": "appointment.read",
            "update appointment": "appointment.update",
            "delete appointment": "appointment.delete",
        },
        "default_action": "profile.read",
    },
    "nurse": {
        "labels": [
            "take test",
            "read vitals", "read media", "delete media",
        ],
        "intent_map": {
            "take test": "takeTest",
            "read vitals": "vital.read",
            "read media": "media.read",
            "delete media": "media.delete",
        },
        "default_action": "vital.read",
    },
    "doctor": {
        "labels": [
            "send message", "open camera", "attach file",
            "create routine", "read routines",
            "update routine", "delete routine",
            "update meal times",
        ],
        "intent_map": {
            "send message": "message.send",
            "open camera": "camera.open",
            "attach file": "message.attach",
            "create routine": "routine.create",
            "read routines": "routine.read",
            "update routine": "routine.update",
            "delete routine": "routine.delete",
            "update meal times": "mealTimes.update",
        },
        "default_action": "routine.read",
    },
}

# ── Entity labels for GLiNER extraction ───────────────────────────────────

ENTITY_LABELS: List[str] = [
    "vital sign type",
    "time reference",
    "frequency",
    "doctor name",
    "person name",
    "body metric",
    "setting name",
    "setting value",
    "location",
    "medication name",
    "symptom or complaint",
    "device name",
    "email address",
    "phone number",
    "meal type",
    "test type",
    "file type",
    "allergen",
]

# ── Entity → field mapping per tool ───────────────────────────────────────

ENTITY_TO_FIELD_MAP: Dict[str, Dict[str, str]] = {
    # ── Agent 1: Receptionist ─────────────────────────────────────────
    "profile.read": {},
    "profile.update": {
        "person name": "name",
        "body metric": "height",
    },
    "camera.open": {},
    "profile.verifyPhone": {"phone number": "otp"},
    "profile.verifyEmail": {"email address": "token"},
    "diagnosis.read": {},
    "allergies.create": {
        "allergen": "name",
        "setting value": "severity",
    },
    "allergies.read": {},
    "allergies.update": {
        "allergen": "name",
        "setting value": "severity",
    },
    "allergies.delete": {"allergen": "name"},
    "careTeam.read": {},
    "family.create": {"email address": "email"},
    "family.read": {},
    "family.update": {"person name": "short_name"},
    "family.delete": {"person name": "member_id"},
    "family.permissionsSchema": {},
    "wifi.read": {},
    "brightness.update": {"setting value": "level"},
    "textSize.update": {"setting value": "size"},
    "device.connect": {"device name": "device_id"},
    "device.read": {},
    "device.delete": {"device name": "device_id"},
    "notification.read": {},
    "notification.update": {
        "setting name": "setting_name",
        "setting value": "value",
    },
    "smartPrompt.update": {"setting value": "enabled"},
    "callSettings.read": {},
    "callSettings.update": {
        "setting name": "setting_name",
        "setting value": "value",
    },
    "appointment.create": {
        "doctor name": "provider_name",
        "time reference": "start_time",
        "location": "location",
        "symptom or complaint": "symptoms",
        "person name": "patient_name",
    },
    "appointment.read": {
        "time reference": "timeframe",
        "doctor name": "provider_name",
    },
    "appointment.update": {
        "doctor name": "provider_name",
        "time reference": "start_time",
    },
    "appointment.delete": {},
    # ── Agent 2: Nurse ────────────────────────────────────────────────
    "takeTest": {
        "vital sign type": "test_type",
        "test type": "test_type",
    },
    "vital.read": {
        "vital sign type": "vital_type",
        "time reference": "timeframe",
    },
    "media.read": {
        "time reference": "timeframe",
        "file type": "media_type",
    },
    "media.delete": {},
    # ── Agent 3: Doctor ───────────────────────────────────────────────
    "message.send": {"person name": "content"},
    "message.attach": {"file type": "file_type"},
    "routine.create": {
        "vital sign type": "type",
        "time reference": "time",
        "frequency": "frequency",
        "medication name": "name",
    },
    "routine.read": {
        "vital sign type": "category",
        "time reference": "timeframe",
    },
    "routine.update": {
        "vital sign type": "type",
        "time reference": "time",
        "frequency": "frequency",
    },
    "routine.delete": {},
    "mealTimes.update": {
        "meal type": "meal",
        "time reference": "time",
    },
}

# ── Keyword fallbacks ─────────────────────────────────────────────────────
# Normal keywords score 1 point each. Priority keywords score 3 points.

AGENT_KEYWORDS: Dict[str, List[str]] = {
    "receptionist": [
        # Profile
        "profile", "my info", "my name", "height", "weight", "avatar",
        "phone", "email", "verify",
        # Conditions
        "allergy", "allergies", "diagnosis", "condition",
        # Care Team
        "care team", "care provider",
        # Family
        "family", "invite", "member", "permissions",
        # Settings
        "setting", "settings", "wifi", "brightness", "text size",
        "font", "notification", "notifications",
        "smart prompt", "call setting", "connected device",
        "dark mode", "device",
        # Appointments
        "doctor", "dr.", "book",
        "clinic", "visit", "checkup",
        "sick", "ill", "unwell", "symptom", "symptoms",
        "fever", "cough", "injury", "feeling",
    ],
    "nurse": [
        "take test", "take my",
        "blood pressure", "bp", "spo2", "blood oxygen", "oxygen",
        "blood glucose", "glucose", "sugar",
        "temperature", "body temperature", "temp",
        "ecg", "ekg", "electrocardiogram",
        "body sounds", "stethoscope", "lungs", "heart sounds",
        "ent", "ear", "nose", "throat",
        "vitals", "vital", "vital sign", "vital signs",
        "reading", "readings", "result", "results",
        "history", "past", "last", "previous",
        "media", "recording", "recordings",
    ],
    "doctor": [
        "health schedule",
        "medication", "medicine", "pill", "tablet",
        "daily", "weekly", "morning", "evening",
        "meal", "meal time", "breakfast", "lunch", "dinner",
        "message", "send", "chat",
        "attach", "attachment", "photo", "picture",
        "remind", "reminder",
    ],
}

# Priority keywords: these score 3x to break agent ties.
# They are exclusively associated with one agent.
AGENT_PRIORITY_KEYWORDS: Dict[str, List[str]] = {
    "receptionist": [
        "appointment", "appointments", "consultation",
    ],
    "nurse": [],
    "doctor": [
        "routine", "routines",
    ],
}

# ── Direct tool-keyword scoring ───────────────────────────────────────────
# Each tool gets verb keywords + topic nouns.
# Tier-2 scores every tool simultaneously rather than verb-type → first label.

TOOL_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    # ── Agent 1: Receptionist ─────────────────────────────────────────
    "profile.read": {
        "verbs": ["show", "view", "see", "get", "display", "what"],
        "nouns": ["profile", "my info", "my name", "my details"],
    },
    "profile.update": {
        "verbs": ["update", "change", "edit", "set", "modify"],
        "nouns": ["profile", "my name", "name", "height", "weight", "avatar"],
    },
    "camera.open": {
        "verbs": ["open", "take", "start"],
        "nouns": ["camera", "photo", "picture", "selfie"],
    },
    "profile.verifyPhone": {
        "verbs": ["verify", "confirm", "validate"],
        "nouns": ["phone", "phone number", "otp"],
    },
    "profile.verifyEmail": {
        "verbs": ["verify", "confirm", "validate"],
        "nouns": ["email", "email address"],
    },
    "diagnosis.read": {
        "verbs": ["show", "view", "see", "get", "what"],
        "nouns": ["diagnosis", "diagnoses", "condition", "conditions"],
    },
    "allergies.create": {
        "verbs": ["add", "create", "new", "log"],
        "nouns": ["allergy", "allergies", "allergic"],
    },
    "allergies.read": {
        "verbs": ["show", "view", "see", "get", "list", "what"],
        "nouns": ["allergy", "allergies", "allergic"],
    },
    "allergies.update": {
        "verbs": ["update", "change", "edit", "modify"],
        "nouns": ["allergy", "allergies"],
    },
    "allergies.delete": {
        "verbs": ["delete", "remove", "clear"],
        "nouns": ["allergy", "allergies"],
    },
    "careTeam.read": {
        "verbs": ["show", "view", "see", "get", "who", "list"],
        "nouns": ["care team", "care provider", "provider", "doctor"],
    },
    "family.create": {
        "verbs": ["add", "invite", "create", "new"],
        "nouns": ["family", "member", "wife", "husband", "child", "parent"],
    },
    "family.read": {
        "verbs": ["show", "view", "see", "get", "list"],
        "nouns": ["family", "family members"],
    },
    "family.update": {
        "verbs": ["update", "change", "edit", "modify"],
        "nouns": ["family", "member"],
    },
    "family.delete": {
        "verbs": ["remove", "delete"],
        "nouns": ["family", "member"],
    },
    "family.permissionsSchema": {
        "verbs": ["show", "view", "what", "get"],
        "nouns": ["permissions", "access"],
    },
    "wifi.read": {
        "verbs": ["show", "view", "check", "get"],
        "nouns": ["wifi", "wi-fi", "network", "internet"],
    },
    "brightness.update": {
        "verbs": ["set", "change", "update", "adjust", "increase", "decrease"],
        "nouns": ["brightness", "screen brightness", "display"],
    },
    "textSize.update": {
        "verbs": ["set", "change", "update", "adjust", "increase", "decrease"],
        "nouns": ["text size", "font size", "font"],
    },
    "device.connect": {
        "verbs": ["connect", "pair", "add", "link"],
        "nouns": ["device", "bluetooth"],
    },
    "device.read": {
        "verbs": ["show", "view", "list", "get", "see"],
        "nouns": ["device", "devices", "connected device", "connected devices"],
    },
    "device.delete": {
        "verbs": ["disconnect", "remove", "delete", "unpair"],
        "nouns": ["device", "devices"],
    },
    "notification.read": {
        "verbs": ["show", "view", "check", "get"],
        "nouns": ["notification", "notifications"],
    },
    "notification.update": {
        "verbs": ["update", "change", "set", "turn", "enable", "disable"],
        "nouns": ["notification", "notifications"],
    },
    "smartPrompt.update": {
        "verbs": ["update", "change", "set", "turn", "enable", "disable"],
        "nouns": ["smart prompt", "prompts"],
    },
    "callSettings.read": {
        "verbs": ["show", "view", "check", "get"],
        "nouns": ["call setting", "call settings", "call"],
    },
    "callSettings.update": {
        "verbs": ["update", "change", "set"],
        "nouns": ["call setting", "call settings", "call"],
    },
    "appointment.create": {
        "verbs": ["create", "book", "schedule", "make", "set up", "new"],
        "nouns": ["appointment", "consultation", "visit", "checkup", "doctor",
                  "sick", "feeling", "ill", "unwell", "symptom"],
    },
    "appointment.read": {
        "verbs": ["show", "view", "see", "get", "list", "check", "what"],
        "nouns": ["appointment", "appointments", "visit", "consultation"],
    },
    "appointment.update": {
        "verbs": ["update", "change", "reschedule", "modify", "edit"],
        "nouns": ["appointment", "visit", "consultation"],
    },
    "appointment.delete": {
        "verbs": ["cancel", "delete", "remove"],
        "nouns": ["appointment", "visit", "consultation"],
    },
    # ── Agent 2: Nurse ────────────────────────────────────────────────
    "takeTest": {
        "verbs": ["take", "do", "perform", "run", "measure", "start", "begin", "record", "want"],
        "nouns": ["test", "blood pressure", "bp", "spo2", "blood oxygen",
                  "blood glucose", "glucose", "ecg", "ekg", "temperature",
                  "body sounds", "ent"],
    },
    "vital.read": {
        "verbs": ["show", "view", "see", "get", "what", "check", "display"],
        "nouns": ["vital", "vitals", "reading", "readings", "result", "results",
                  "history", "blood pressure", "bp", "spo2", "glucose",
                  "ecg", "temperature", "last", "past", "previous"],
    },
    "media.read": {
        "verbs": ["show", "view", "see", "get", "list", "play"],
        "nouns": ["media", "recording", "recordings", "audio", "video"],
    },
    "media.delete": {
        "verbs": ["delete", "remove", "clear"],
        "nouns": ["media", "recording", "recordings"],
    },
    # ── Agent 3: Doctor ───────────────────────────────────────────────
    "message.send": {
        "verbs": ["send", "write", "text", "tell", "contact"],
        "nouns": ["message", "doctor", "chat"],
    },
    "message.attach": {
        "verbs": ["attach", "upload", "share", "send"],
        "nouns": ["file", "photo", "picture", "attachment", "document"],
    },
    "routine.create": {
        "verbs": ["create", "add", "new", "set up", "start", "schedule", "make"],
        "nouns": ["routine", "routines", "schedule", "reminder",
                  "medication", "medicine", "pill"],
    },
    "routine.read": {
        "verbs": ["show", "view", "see", "get", "list", "what", "check"],
        "nouns": ["routine", "routines", "schedule", "health schedule"],
    },
    "routine.update": {
        "verbs": ["update", "change", "edit", "modify"],
        "nouns": ["routine", "routines", "schedule"],
    },
    "routine.delete": {
        "verbs": ["delete", "remove", "cancel", "stop"],
        "nouns": ["routine", "routines", "schedule"],
    },
    "mealTimes.update": {
        "verbs": ["set", "change", "update", "adjust"],
        "nouns": ["meal", "meal time", "breakfast", "lunch", "dinner"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Intent Extractor
# ═══════════════════════════════════════════════════════════════════════════

class IntentExtractor:
    """GLiNER-powered NLU with keyword fallback."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from gliner import GLiNER
            logger.info("Loading GLiNER model: %s", self.model_name)
            self._model = GLiNER.from_pretrained(self.model_name)
        return self._model

    def _gliner_predict(
        self, text: str, labels: List[str], threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        model = self._load_model()
        try:
            return model.predict_entities(text, labels, threshold=threshold)
        except Exception as e:
            logger.warning("GLiNER predict failed: %s", e)
            return []

    # ── Tier 1: Agent classification ──────────────────────────────────

    def _classify_agent_gliner(self, text: str) -> tuple[Optional[str], float]:
        preds = self._gliner_predict(text, AGENT_LABELS, 0.15)
        if not preds:
            return None, 0.0
        best = max(preds, key=lambda p: p["score"])
        agent = AGENT_MAP.get(best["label"])
        return agent, best["score"]

    def _classify_agent_keywords(self, text: str) -> Optional[str]:
        lower = text.lower()
        scores: Dict[str, float] = {}
        for agent, keywords in AGENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in lower)
            # Priority keywords score 3x
            priority_score = sum(
                3 for kw in AGENT_PRIORITY_KEYWORDS.get(agent, []) if kw in lower
            )
            total = score + priority_score
            if total > 0:
                scores[agent] = total
        if not scores:
            return None
        return max(scores, key=scores.get)

    # ── Tier 2: Tool classification (topic-aware) ─────────────────────

    def _classify_tool_hybrid(
        self, text: str, agent: str
    ) -> tuple[Optional[str], float]:
        """Score every tool by verb+noun keyword match, then blend with GLiNER."""
        config = SUB_ACTION_CONFIG.get(agent)
        if not config:
            return None, 0.0

        lower = text.lower()
        valid_tools = set(config["intent_map"].values())

        # ── Keyword scoring per tool ──────────────────────────────
        tool_scores: Dict[str, float] = {}
        for tool_name, kw_config in TOOL_KEYWORDS.items():
            if tool_name not in valid_tools:
                continue
            verb_hits = sum(1 for v in kw_config["verbs"] if v in lower)
            noun_hits = sum(1 for n in kw_config["nouns"] if n in lower)
            # Nouns are more important than verbs (weight 2x)
            score = verb_hits + (noun_hits * 2)
            if score > 0:
                tool_scores[tool_name] = score

        best_kw_tool = max(tool_scores, key=tool_scores.get) if tool_scores else None
        best_kw_score = tool_scores.get(best_kw_tool, 0) if best_kw_tool else 0

        # ── GLiNER scoring ────────────────────────────────────────
        preds = self._gliner_predict(text, config["labels"], 0.15)
        gliner_action = None
        gliner_conf = 0.0
        if preds:
            best = max(preds, key=lambda p: p["score"])
            gliner_action = config["intent_map"].get(best["label"])
            gliner_conf = best["score"]

        logger.info(
            "Tier-2 hybrid: gliner=%r(%.3f)  kw_best=%r(%.1f)",
            gliner_action, gliner_conf, best_kw_tool, best_kw_score,
        )

        # ── Decision logic ────────────────────────────────────────
        # Keywords with strong signal (noun match = 2+ score) override GLiNER
        if best_kw_score >= 3 and best_kw_tool != gliner_action:
            logger.info(
                "Keywords override: %r (score=%.1f) beats %r (conf=%.3f)",
                best_kw_tool, best_kw_score, gliner_action, gliner_conf,
            )
            return best_kw_tool, gliner_conf

        # Confident GLiNER
        if gliner_action and gliner_conf >= 0.3:
            return gliner_action, gliner_conf

        # Fall back to keyword best
        if best_kw_tool:
            return best_kw_tool, gliner_conf

        return config.get("default_action"), 0.0

    # ── Entity extraction ─────────────────────────────────────────────

    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
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

    # ── Symptom heuristic ─────────────────────────────────────────────

    @staticmethod
    def _extract_symptoms_heuristic(text: str) -> Optional[str]:
        """Extract symptoms from conversational patterns."""
        lower = text.lower()
        symptom_patterns = [
            r"i\s+(?:don'?t|can'?t|couldn'?t|cannot|am not able to)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
            r"i(?:'?m| am)\s+(?:feeling|having|experiencing)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
            r"i\s+(?:have|feel|experience|notice|get|got)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
            r"(?:suffering|troubled|bothered)\s+(?:from|by|with)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
            r"my\s+(.+?\s+(?:hurts?|aches?|is\s+(?:swollen|sore|painful|stiff|numb)))",
            r"(?:pain|ache|problem|issue|trouble|difficulty)\s+(?:in|with)\s+(.+?)(?:,|\.|$|create|book|schedule|make|set)",
        ]
        for pattern in symptom_patterns:
            match = re.search(pattern, lower)
            if match:
                symptom = match.group(1).strip().rstrip(",. ")
                if len(symptom) > 2:
                    logger.info("Heuristic symptom: %r", symptom)
                    return symptom
        return None

    # ── Slot filling ──────────────────────────────────────────────────

    def _fill_slots(
        self, action: str, entities: List[ExtractedEntity], text: str
    ) -> tuple[Dict[str, Any], List[str]]:
        field_map = ENTITY_TO_FIELD_MAP.get(action, {})
        model_cls = TOOL_REGISTRY.get(action)

        filled: Dict[str, Any] = {}
        for ent in entities:
            field_name = field_map.get(ent.label)
            if field_name and field_name not in filled:
                value = ent.text
                # Clean allergen text: "nuts allergy" → "nuts"
                if ent.label == "allergen":
                    import re
                    value = re.sub(r"\s*\b(allergy|allergies|allergic)\b\s*", "", value, flags=re.IGNORECASE).strip()
                filled[field_name] = value

        # Heuristic symptom fallback for appointment actions
        if action in ("appointment.create", "appointment.read") and "symptoms" not in filled:
            symptom = self._extract_symptoms_heuristic(text)
            if symptom:
                filled["symptoms"] = symptom

        # Check missing required fields
        missing: List[str] = []
        if model_cls:
            try:
                instance = model_cls(**filled)
                missing = instance.get_missing_required()
            except Exception:
                missing = getattr(model_cls, "REQUIRED_FIELDS", [])

        return filled, missing

    # ── Main pipeline ─────────────────────────────────────────────────

    def analyse(self, text: str) -> PipelineResult:
        result = PipelineResult(raw_text=text)

        # ── Tier 1: Agent classification (hybrid) ─────────────────
        gliner_agent, agent_conf = self._classify_agent_gliner(text)
        kw_agent = self._classify_agent_keywords(text)
        logger.info(
            "Tier-1: gliner=%r(%.3f)  keywords=%r",
            gliner_agent, agent_conf, kw_agent,
        )

        # Always prefer keywords when they disagree with GLiNER
        # (priority keywords ensure strong signal for appointment/routine)
        if kw_agent and kw_agent != gliner_agent:
            agent = kw_agent
            logger.info("Tier-1: keywords override GLiNER → %r", agent)
        elif gliner_agent and agent_conf >= DOMAIN_THRESHOLD:
            agent = gliner_agent
        elif kw_agent:
            agent = kw_agent
        else:
            agent = gliner_agent

        if agent is None:
            logger.info("No agent classified — returning fallback result.")
            return result

        result.agent = agent
        result.confidence = agent_conf

        # ── Tier 2: Tool classification (hybrid) ──────────────────
        action, action_conf = self._classify_tool_hybrid(text, agent)
        logger.info("Tier-2 result: action=%r  confidence=%.3f", action, action_conf)

        if action is None:
            action = SUB_ACTION_CONFIG[agent].get("default_action")
            logger.info("Using default action for agent %r: %r", agent, action)

        result.action = action
        result.tool_name = action

        # ── Entity extraction + slot filling ──────────────────────
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
