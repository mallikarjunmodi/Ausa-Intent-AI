"""
schemas.py — Pydantic Data Models for the Health Pipeline

Defines the data contracts between the NLU pipeline (which fills fields
from voice commands) and the backend/database layer.

Domains:
  • Routines     — create / view / update / delete scheduled health activities
  • Profiles     — view / update user profile info
  • Appointments — create / view / cancel doctor appointments
  • Settings     — change / view device settings
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class VitalTestType(str, Enum):
    """Types of vital-sign tests the device can perform."""
    BLOOD_PRESSURE = "blood_pressure"
    SPO2 = "spo2"
    HEART_RATE = "heart_rate"
    ECG = "ecg"
    TEMPERATURE = "temperature"
    BLOOD_GLUCOSE = "blood_glucose"
    WEIGHT = "weight"
    GENERIC = "generic"


class FrequencyType(str, Enum):
    """How often a routine repeats."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ONCE = "once"
    INTERVAL = "interval"


class AllergySeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


# ═══════════════════════════════════════════════════════════════════════════
# Routines Domain
# ═══════════════════════════════════════════════════════════════════════════

class RoutineCreateRequest(BaseModel):
    """Payload to create a new health routine.

    Required fields the NLU must fill (or prompt the user for):
      - routine_name
      - vital_test_type
      - scheduled_time
    """
    routine_name: Optional[str] = Field(None, description="Name of the routine")
    vital_test_type: Optional[str] = Field(None, description="Type of vital test (e.g. blood pressure, heart rate)")
    frequency: Optional[str] = Field(None, description="How often: daily, weekly, monthly, once")
    scheduled_time: Optional[str] = Field(None, description="When to perform (e.g. '8 AM', 'morning', 'every 4 hours')")
    notes: Optional[str] = Field(None, description="Additional notes")

    REQUIRED_FIELDS: List[str] = ["vital_test_type", "scheduled_time"]

    class Config:
        # Allow REQUIRED_FIELDS as class attr without Pydantic complaining
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        """Return names of required fields that are still None."""
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


class QueryRoutinesArgs(BaseModel):
    """Arguments to query/filter existing routines."""
    search_query: Optional[str] = Field(None, description="Free-text search")
    category: Optional[str] = Field(None, description="Filter by vital test type")
    frequency: Optional[str] = Field(None, description="Filter by frequency")
    date_from: Optional[str] = Field(None, description="Start of date range")
    date_to: Optional[str] = Field(None, description="End of date range")
    timeframe: Optional[str] = Field(None, description="Natural language timeframe (e.g. 'this week')")

    REQUIRED_FIELDS: List[str] = []

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


class ViewResultArgs(BaseModel):
    """View past vital sign readings/results."""
    vital_type: Optional[str] = Field(None, description="Type of vital to view (e.g. 'blood pressure', 'heart rate')")
    timeframe: Optional[str] = Field(None, description="Time range (e.g. 'last', 'past week', 'today')")

    REQUIRED_FIELDS: List[str] = []

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


class UpdateRoutineArgs(BaseModel):
    """Patch model to update fields on an existing routine."""
    routine_id: Optional[str] = Field(None, description="ID of routine to update")
    routine_name: Optional[str] = Field(None, description="New name")
    vital_test_type: Optional[str] = Field(None, description="New vital test type")
    frequency: Optional[str] = Field(None, description="New frequency")
    scheduled_time: Optional[str] = Field(None, description="New scheduled time")
    notes: Optional[str] = Field(None, description="New notes")

    REQUIRED_FIELDS: List[str] = ["routine_id"]

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


class DeleteRoutineArgs(BaseModel):
    """Delete a routine by ID."""
    routine_id: Optional[str] = Field(None, description="ID of routine to delete")
    routine_name: Optional[str] = Field(None, description="Name of routine to delete")
    hard_delete: bool = Field(False, description="Permanently delete vs soft-delete")

    REQUIRED_FIELDS: List[str] = ["routine_id"]

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


# ═══════════════════════════════════════════════════════════════════════════
# Profiles Domain
# ═══════════════════════════════════════════════════════════════════════════

class ProfileViewRequest(BaseModel):
    """View user profile — no required fields."""
    section: Optional[str] = Field(None, description="Specific section to view (e.g. 'allergies', 'care team')")

    REQUIRED_FIELDS: List[str] = []

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


class ProfileUpdateRequest(BaseModel):
    """Update user profile fields."""
    name: Optional[str] = Field(None, description="User's name")
    height: Optional[str] = Field(None, description="Height (e.g. '180 cm', '5 foot 11')")
    weight: Optional[str] = Field(None, description="Weight (e.g. '75 kg', '165 lbs')")
    avatar: Optional[str] = Field(None, description="Avatar preference")

    REQUIRED_FIELDS: List[str] = []

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


# ═══════════════════════════════════════════════════════════════════════════
# Appointments Domain
# ═══════════════════════════════════════════════════════════════════════════

class AppointmentCreateRequest(BaseModel):
    """Create a new appointment."""
    doctor_name: Optional[str] = Field(None, description="Doctor's name")
    specialty: Optional[str] = Field(None, description="Doctor's specialty")
    date_time: Optional[str] = Field(None, description="Appointment date/time")
    location: Optional[str] = Field(None, description="Location or clinic name")
    symptoms: Optional[str] = Field(None, description="Symptoms or complaints")
    patient_name: Optional[str] = Field(None, description="Patient's name")
    notes: Optional[str] = Field(None, description="Additional notes")

    REQUIRED_FIELDS: List[str] = ["doctor_name", "date_time"]

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


class AppointmentViewRequest(BaseModel):
    """View/query appointments."""
    timeframe: Optional[str] = Field(None, description="Time range (e.g. 'this week', 'next month')")
    doctor_name: Optional[str] = Field(None, description="Filter by doctor")

    REQUIRED_FIELDS: List[str] = []

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


class AppointmentCancelRequest(BaseModel):
    """Cancel an existing appointment."""
    appointment_id: Optional[str] = Field(None, description="Appointment ID to cancel")
    doctor_name: Optional[str] = Field(None, description="Doctor name (to identify appointment)")
    date_time: Optional[str] = Field(None, description="Date/time (to identify appointment)")

    REQUIRED_FIELDS: List[str] = ["appointment_id"]

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


# ═══════════════════════════════════════════════════════════════════════════
# Settings Domain
# ═══════════════════════════════════════════════════════════════════════════

class SettingsUpdateRequest(BaseModel):
    """Change a device/app setting."""
    setting_name: Optional[str] = Field(None, description="Which setting (e.g. 'dark mode', 'brightness', 'notifications')")
    setting_value: Optional[str] = Field(None, description="New value (e.g. 'on', '80%', 'enabled')")

    REQUIRED_FIELDS: List[str] = ["setting_name"]

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


class SettingsViewRequest(BaseModel):
    """View current settings."""
    setting_name: Optional[str] = Field(None, description="Specific setting to view, or None for all")

    REQUIRED_FIELDS: List[str] = []

    class Config:
        fields = {"REQUIRED_FIELDS": {"exclude": True}}

    def get_missing_required(self) -> List[str]:
        return [f for f in self.REQUIRED_FIELDS if getattr(self, f) is None]


# ═══════════════════════════════════════════════════════════════════════════
# Tool Registry — maps action names to their Pydantic model class
# ═══════════════════════════════════════════════════════════════════════════

TOOL_REGISTRY: Dict[str, type] = {
    # Routines
    "create_routine": RoutineCreateRequest,
    "view_routines": QueryRoutinesArgs,
    "view_result": ViewResultArgs,
    "update_routine": UpdateRoutineArgs,
    "delete_routine": DeleteRoutineArgs,
    # Profiles
    "view_profile": ProfileViewRequest,
    "update_profile": ProfileUpdateRequest,
    # Appointments
    "create_appointment": AppointmentCreateRequest,
    "view_appointments": AppointmentViewRequest,
    "cancel_appointment": AppointmentCancelRequest,
    # Settings
    "change_setting": SettingsUpdateRequest,
    "view_settings": SettingsViewRequest,
}
