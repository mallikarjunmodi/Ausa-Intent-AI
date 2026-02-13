"""
schemas.py — Pydantic Data Models for Hierarchical Agent System

Three-agent hierarchy mirroring real healthcare workflows:

  Agent 1 — Health Management (Receptionist)
      Profile, Conditions, Care Team, Family, Settings, Appointments

  Agent 2 — Health Recording (Nurse)
      Take Tests, Vitals History, Media History

  Agent 3 — Health Consultation (Doctor)
      Messaging, Health Schedule (Routines, Meal Times)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class TestType(str, Enum):
    BLOOD_PRESSURE = "blood_pressure"
    BLOOD_OXYGEN = "blood_oxygen"
    BLOOD_GLUCOSE = "blood_glucose"
    BODY_TEMPERATURE = "body_temperature"
    ECG = "ecg"
    BODY_SOUNDS = "body_sounds"
    ENT = "ent"


class AllergySeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class FamilyInviteMethod(str, Enum):
    EMAIL_INVITE = "emailInvite"
    QR = "qr"


class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"


class CameraSource(str, Enum):
    HUB = "hub"
    AUSA_X = "ausa_x"


# ═══════════════════════════════════════════════════════════════════════════
# Base class with required-field tracking
# ═══════════════════════════════════════════════════════════════════════════

class ToolModel(BaseModel):
    """Base for all tool argument models."""
    REQUIRED_FIELDS: List[str] = []

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]

    class Config:
        # Prevent Pydantic from treating REQUIRED_FIELDS as a model field
        json_schema_extra = None


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1 — Health Management (Receptionist)
# ═══════════════════════════════════════════════════════════════════════════

# ── Profile ───────────────────────────────────────────────────────────────

class ProfileReadArgs(ToolModel):
    section: Optional[str] = Field(None, description="Section to view (e.g. 'personal', 'contact')")
    REQUIRED_FIELDS: List[str] = []


class ProfileUpdateArgs(ToolModel):
    name: Optional[str] = Field(None, description="User's name")
    height: Optional[str] = Field(None, description="Height (e.g. '180 cm')")
    weight: Optional[str] = Field(None, description="Weight (e.g. '75 kg')")
    avatar: Optional[str] = Field(None, description="Avatar preference")
    REQUIRED_FIELDS: List[str] = []


class CameraOpenArgs(ToolModel):
    source: Optional[str] = Field(None, description="Camera source: hub or ausa_x")
    REQUIRED_FIELDS: List[str] = []


class ProfileVerifyPhoneArgs(ToolModel):
    otp: Optional[str] = Field(None, description="One-time password for phone verification")
    REQUIRED_FIELDS: List[str] = ["otp"]


class ProfileVerifyEmailArgs(ToolModel):
    token: Optional[str] = Field(None, description="Token for email verification")
    REQUIRED_FIELDS: List[str] = ["token"]


# ── Conditions ────────────────────────────────────────────────────────────

class DiagnosisReadArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


class AllergiesCreateArgs(ToolModel):
    name: Optional[str] = Field(None, description="Allergy name")
    severity: Optional[str] = Field(None, description="Severity: low, moderate, high, severe")
    notes: Optional[str] = Field(None, description="Additional notes")
    REQUIRED_FIELDS: List[str] = ["name"]


class AllergiesReadArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


class AllergiesUpdateArgs(ToolModel):
    allergy_id: Optional[str] = Field(None, description="ID of allergy to update")
    name: Optional[str] = Field(None, description="New name")
    severity: Optional[str] = Field(None, description="New severity")
    notes: Optional[str] = Field(None, description="New notes")
    REQUIRED_FIELDS: List[str] = ["allergy_id"]


class AllergiesDeleteArgs(ToolModel):
    allergy_id: Optional[str] = Field(None, description="ID of allergy to delete")
    REQUIRED_FIELDS: List[str] = ["allergy_id"]


# ── Care Team ─────────────────────────────────────────────────────────────

class CareTeamReadArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


# ── Family ────────────────────────────────────────────────────────────────

class FamilyCreateArgs(ToolModel):
    email: Optional[str] = Field(None, description="Family member's email")
    via: Optional[str] = Field(None, description="Invite method: emailInvite or qr")
    REQUIRED_FIELDS: List[str] = ["email"]


class FamilyReadArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


class FamilyUpdateArgs(ToolModel):
    member_id: Optional[str] = Field(None, description="Family member ID")
    short_name: Optional[str] = Field(None, description="Short name / nickname")
    relation: Optional[str] = Field(None, description="Relationship (e.g. spouse, child)")
    permissions: Optional[str] = Field(None, description="Permissions to grant")
    REQUIRED_FIELDS: List[str] = ["member_id"]


class FamilyDeleteArgs(ToolModel):
    member_id: Optional[str] = Field(None, description="Family member ID to remove")
    REQUIRED_FIELDS: List[str] = ["member_id"]


class FamilyPermissionsSchemaArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


# ── Settings ──────────────────────────────────────────────────────────────

class WifiReadArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


class BrightnessUpdateArgs(ToolModel):
    level: Optional[str] = Field(None, description="Brightness level (e.g. '80%', 'high')")
    REQUIRED_FIELDS: List[str] = ["level"]


class TextSizeUpdateArgs(ToolModel):
    size: Optional[str] = Field(None, description="Text size (e.g. 'large', 'small', '16px')")
    REQUIRED_FIELDS: List[str] = ["size"]


class DeviceConnectArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


class DeviceReadArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


class DeviceDeleteArgs(ToolModel):
    device_id: Optional[str] = Field(None, description="Device ID to disconnect")
    REQUIRED_FIELDS: List[str] = ["device_id"]


class NotificationReadArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


class NotificationUpdateArgs(ToolModel):
    setting_name: Optional[str] = Field(None, description="Notification setting name")
    value: Optional[str] = Field(None, description="New value (e.g. 'on', 'off')")
    REQUIRED_FIELDS: List[str] = ["setting_name"]


class SmartPromptUpdateArgs(ToolModel):
    enabled: Optional[str] = Field(None, description="Enable or disable: 'on' or 'off'")
    REQUIRED_FIELDS: List[str] = ["enabled"]


class CallSettingsReadArgs(ToolModel):
    REQUIRED_FIELDS: List[str] = []


class CallSettingsUpdateArgs(ToolModel):
    setting_name: Optional[str] = Field(None, description="Call setting to change")
    value: Optional[str] = Field(None, description="New value")
    REQUIRED_FIELDS: List[str] = ["setting_name"]


# ── Appointments ──────────────────────────────────────────────────────────

class AppointmentCreateArgs(ToolModel):
    provider_name: Optional[str] = Field(None, description="Doctor/provider name")
    start_time: Optional[str] = Field(None, description="Appointment start time")
    end_time: Optional[str] = Field(None, description="Appointment end time")
    symptoms: Optional[str] = Field(None, description="Symptoms or complaints")
    patient_name: Optional[str] = Field(None, description="Patient name")
    REQUIRED_FIELDS: List[str] = ["provider_name", "start_time"]


class AppointmentReadArgs(ToolModel):
    timeframe: Optional[str] = Field(None, description="Time range (e.g. 'this week')")
    provider_name: Optional[str] = Field(None, description="Filter by provider")
    REQUIRED_FIELDS: List[str] = []


class AppointmentUpdateArgs(ToolModel):
    target_id: Optional[str] = Field(None, description="Appointment ID to update")
    provider_name: Optional[str] = Field(None, description="New provider name")
    start_time: Optional[str] = Field(None, description="New start time")
    end_time: Optional[str] = Field(None, description="New end time")
    REQUIRED_FIELDS: List[str] = ["target_id"]


class AppointmentDeleteArgs(ToolModel):
    target_id: Optional[str] = Field(None, description="Appointment ID to cancel")
    REQUIRED_FIELDS: List[str] = ["target_id"]


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2 — Health Recording (Nurse)
# ═══════════════════════════════════════════════════════════════════════════

class TakeTestArgs(ToolModel):
    test_type: Optional[str] = Field(None, description="Test: blood_pressure, blood_oxygen, blood_glucose, body_temperature, ecg, body_sounds, ent")
    REQUIRED_FIELDS: List[str] = ["test_type"]


class VitalReadArgs(ToolModel):
    vital_type: Optional[str] = Field(None, description="Type of vital to view")
    timeframe: Optional[str] = Field(None, description="Time range (e.g. 'last week')")
    REQUIRED_FIELDS: List[str] = []


class MediaReadArgs(ToolModel):
    media_type: Optional[str] = Field(None, description="Type of media (e.g. 'ecg', 'body_sounds')")
    timeframe: Optional[str] = Field(None, description="Time range")
    REQUIRED_FIELDS: List[str] = []


class MediaDeleteArgs(ToolModel):
    media_id: Optional[str] = Field(None, description="Media ID to delete")
    REQUIRED_FIELDS: List[str] = ["media_id"]


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 3 — Health Consultation (Doctor)
# ═══════════════════════════════════════════════════════════════════════════

class MessageSendArgs(ToolModel):
    content: Optional[str] = Field(None, description="Message content")
    REQUIRED_FIELDS: List[str] = ["content"]


class MessageAttachArgs(ToolModel):
    file_type: Optional[str] = Field(None, description="Type of file to attach")
    REQUIRED_FIELDS: List[str] = []


class RoutineCreateArgs(ToolModel):
    name: Optional[str] = Field(None, description="Routine name")
    type: Optional[str] = Field(None, description="Routine type (e.g. vital test, medication)")
    frequency: Optional[str] = Field(None, description="How often: daily, weekly, etc.")
    time: Optional[str] = Field(None, description="Scheduled time")
    duration: Optional[str] = Field(None, description="Duration of routine")
    REQUIRED_FIELDS: List[str] = ["name", "time"]


class RoutineReadArgs(ToolModel):
    category: Optional[str] = Field(None, description="Filter by category")
    timeframe: Optional[str] = Field(None, description="Time range")
    REQUIRED_FIELDS: List[str] = []


class RoutineUpdateArgs(ToolModel):
    routine_id: Optional[str] = Field(None, description="Routine ID to update")
    name: Optional[str] = Field(None, description="New name")
    type: Optional[str] = Field(None, description="New type")
    frequency: Optional[str] = Field(None, description="New frequency")
    time: Optional[str] = Field(None, description="New time")
    REQUIRED_FIELDS: List[str] = ["routine_id"]


class RoutineDeleteArgs(ToolModel):
    routine_id: Optional[str] = Field(None, description="Routine ID to delete")
    REQUIRED_FIELDS: List[str] = ["routine_id"]


class MealTimesUpdateArgs(ToolModel):
    meal: Optional[str] = Field(None, description="Meal: breakfast, lunch, or dinner")
    time: Optional[str] = Field(None, description="New meal time")
    REQUIRED_FIELDS: List[str] = ["meal", "time"]


# ═══════════════════════════════════════════════════════════════════════════
# Tool Registry — maps tool names to Pydantic model classes
# ═══════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, type] = {
    # ── Agent 1: Receptionist ─────────────────────────────────────────
    # Profile
    "profile.read": ProfileReadArgs,
    "profile.update": ProfileUpdateArgs,
    "camera.open": CameraOpenArgs,
    "profile.verifyPhone": ProfileVerifyPhoneArgs,
    "profile.verifyEmail": ProfileVerifyEmailArgs,
    # Conditions
    "diagnosis.read": DiagnosisReadArgs,
    "allergies.create": AllergiesCreateArgs,
    "allergies.read": AllergiesReadArgs,
    "allergies.update": AllergiesUpdateArgs,
    "allergies.delete": AllergiesDeleteArgs,
    # Care Team
    "careTeam.read": CareTeamReadArgs,
    # Family
    "family.create": FamilyCreateArgs,
    "family.read": FamilyReadArgs,
    "family.update": FamilyUpdateArgs,
    "family.delete": FamilyDeleteArgs,
    "family.permissionsSchema": FamilyPermissionsSchemaArgs,
    # Settings
    "wifi.read": WifiReadArgs,
    "brightness.update": BrightnessUpdateArgs,
    "textSize.update": TextSizeUpdateArgs,
    "device.connect": DeviceConnectArgs,
    "device.read": DeviceReadArgs,
    "device.delete": DeviceDeleteArgs,
    "notification.read": NotificationReadArgs,
    "notification.update": NotificationUpdateArgs,
    "smartPrompt.update": SmartPromptUpdateArgs,
    "callSettings.read": CallSettingsReadArgs,
    "callSettings.update": CallSettingsUpdateArgs,
    # Appointments
    "appointment.create": AppointmentCreateArgs,
    "appointment.read": AppointmentReadArgs,
    "appointment.update": AppointmentUpdateArgs,
    "appointment.delete": AppointmentDeleteArgs,
    # ── Agent 2: Nurse ────────────────────────────────────────────────
    "takeTest": TakeTestArgs,
    "vital.read": VitalReadArgs,
    "media.read": MediaReadArgs,
    "media.delete": MediaDeleteArgs,
    # ── Agent 3: Doctor ───────────────────────────────────────────────
    "message.send": MessageSendArgs,
    "message.attach": MessageAttachArgs,
    "routine.create": RoutineCreateArgs,
    "routine.read": RoutineReadArgs,
    "routine.update": RoutineUpdateArgs,
    "routine.delete": RoutineDeleteArgs,
    "mealTimes.update": MealTimesUpdateArgs,
}
