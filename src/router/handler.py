"""
handler.py — Agent-based router for the Health Pipeline

Dispatches PipelineResult to the appropriate mock handler based on
the classified tool name. Groups handlers by agent for clarity.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from src.nlu.extractor import PipelineResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Agent 1 — Health Management (Receptionist)
# ═══════════════════════════════════════════════════════════════════════════

# ── Profile ───────────────────────────────────────────────────────────────

def mock_profile_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("✅  PROFILE  ➜  READ")
    print("-" * 60)
    print(f"  Section : {args.get('section', '<all>')}")
    print("  ▸ Loading profile data …")
    print("=" * 60 + "\n")


def mock_profile_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("✅  PROFILE  ➜  UPDATE")
    print("-" * 60)
    for k, v in args.items():
        print(f"  {k:12s} = {v!r}")
    print("  ▸ Saving profile changes …")
    print("=" * 60 + "\n")


def mock_camera_open(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📷  CAMERA  ➜  OPEN")
    print("-" * 60)
    print(f"  Source : {args.get('source', '<default>')}")
    print("  ▸ Opening camera …")
    print("=" * 60 + "\n")


def mock_verify_phone(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📱  PROFILE  ➜  VERIFY PHONE")
    print("-" * 60)
    print(f"  OTP : {args.get('otp', '<not provided>')}")
    print("  ▸ Verifying phone number …")
    print("=" * 60 + "\n")


def mock_verify_email(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📧  PROFILE  ➜  VERIFY EMAIL")
    print("-" * 60)
    print(f"  Token : {args.get('token', '<not provided>')}")
    print("  ▸ Verifying email address …")
    print("=" * 60 + "\n")


# ── Conditions ────────────────────────────────────────────────────────────

def mock_diagnosis_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🩺  CONDITION  ➜  READ DIAGNOSIS")
    print("-" * 60)
    print("  ▸ Loading diagnosis records …")
    print("=" * 60 + "\n")


def mock_allergies_create(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🤧  ALLERGY  ➜  CREATE")
    print("-" * 60)
    print(f"  Name     : {args.get('name', '<not specified>')}")
    print(f"  Severity : {args.get('severity', '<not specified>')}")
    print(f"  Notes    : {args.get('notes', '<none>')}")
    print("  ▸ Saving new allergy …")
    print("=" * 60 + "\n")


def mock_allergies_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🤧  ALLERGY  ➜  READ")
    print("-" * 60)
    print("  ▸ Loading allergy list …")
    print("=" * 60 + "\n")


def mock_allergies_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🤧  ALLERGY  ➜  UPDATE")
    print("-" * 60)
    for k, v in args.items():
        print(f"  {k:12s} = {v!r}")
    print("  ▸ Updating allergy …")
    print("=" * 60 + "\n")


def mock_allergies_delete(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🤧  ALLERGY  ➜  DELETE")
    print("-" * 60)
    print(f"  ID/Name : {args.get('allergy_id', args.get('name', '<unknown>'))}")
    print("  ▸ Removing allergy …")
    print("=" * 60 + "\n")


# ── Care Team ─────────────────────────────────────────────────────────────

def mock_careteam_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("👥  CARE TEAM  ➜  READ")
    print("-" * 60)
    print("  ▸ Loading care team …")
    print("=" * 60 + "\n")


# ── Family ────────────────────────────────────────────────────────────────

def mock_family_create(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("👨‍👩‍👧  FAMILY  ➜  ADD MEMBER")
    print("-" * 60)
    print(f"  Email : {args.get('email', '<not specified>')}")
    print(f"  Via   : {args.get('via', 'emailInvite')}")
    print("  ▸ Sending invite …")
    print("=" * 60 + "\n")


def mock_family_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("👨‍👩‍👧  FAMILY  ➜  READ")
    print("-" * 60)
    print("  ▸ Loading family members …")
    print("=" * 60 + "\n")


def mock_family_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("👨‍👩‍👧  FAMILY  ➜  UPDATE MEMBER")
    print("-" * 60)
    for k, v in args.items():
        print(f"  {k:12s} = {v!r}")
    print("  ▸ Updating member …")
    print("=" * 60 + "\n")


def mock_family_delete(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("👨‍👩‍👧  FAMILY  ➜  REMOVE MEMBER")
    print("-" * 60)
    print(f"  Member : {args.get('member_id', '<unknown>')}")
    print("  ▸ Removing family member …")
    print("=" * 60 + "\n")


def mock_family_permissions(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🔐  FAMILY  ➜  PERMISSIONS SCHEMA")
    print("-" * 60)
    print("  Available: Health Schedule, Appointments, Vitals History")
    print("=" * 60 + "\n")


# ── Settings ──────────────────────────────────────────────────────────────

def mock_wifi_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📶  SETTINGS  ➜  WIFI STATUS")
    print("-" * 60)
    print("  ▸ Reading WiFi settings …")
    print("=" * 60 + "\n")


def mock_brightness_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🔆  SETTINGS  ➜  BRIGHTNESS")
    print("-" * 60)
    print(f"  Level : {args.get('level', '<not specified>')}")
    print("  ▸ Adjusting brightness …")
    print("=" * 60 + "\n")


def mock_textsize_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🔤  SETTINGS  ➜  TEXT SIZE")
    print("-" * 60)
    print(f"  Size : {args.get('size', '<not specified>')}")
    print("  ▸ Adjusting text size …")
    print("=" * 60 + "\n")


def mock_device_connect(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🔌  SETTINGS  ➜  CONNECT DEVICE")
    print("-" * 60)
    print("  ▸ Scanning for devices …")
    print("=" * 60 + "\n")


def mock_device_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🔌  SETTINGS  ➜  CONNECTED DEVICES")
    print("-" * 60)
    print("  ▸ Loading device list …")
    print("=" * 60 + "\n")


def mock_device_delete(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🔌  SETTINGS  ➜  DISCONNECT DEVICE")
    print("-" * 60)
    print(f"  Device : {args.get('device_id', '<unknown>')}")
    print("  ▸ Disconnecting …")
    print("=" * 60 + "\n")


def mock_notification_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🔔  SETTINGS  ➜  NOTIFICATIONS")
    print("-" * 60)
    print("  ▸ Loading notification settings …")
    print("=" * 60 + "\n")


def mock_notification_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🔔  SETTINGS  ➜  UPDATE NOTIFICATION")
    print("-" * 60)
    print(f"  Setting : {args.get('setting_name', '<not specified>')}")
    print(f"  Value   : {args.get('value', '<not specified>')}")
    print("  ▸ Saving notification settings …")
    print("=" * 60 + "\n")


def mock_smart_prompt_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("💡  SETTINGS  ➜  SMART PROMPT")
    print("-" * 60)
    print(f"  Enabled : {args.get('enabled', '<not specified>')}")
    print("  ▸ Updating smart prompt …")
    print("=" * 60 + "\n")


def mock_call_settings_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📞  SETTINGS  ➜  CALL SETTINGS")
    print("-" * 60)
    print("  ▸ Loading call settings …")
    print("=" * 60 + "\n")


def mock_call_settings_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📞  SETTINGS  ➜  UPDATE CALL SETTINGS")
    print("-" * 60)
    print(f"  Setting : {args.get('setting_name', '<not specified>')}")
    print(f"  Value   : {args.get('value', '<not specified>')}")
    print("  ▸ Saving call settings …")
    print("=" * 60 + "\n")


# ── Appointments ──────────────────────────────────────────────────────────

def mock_appointment_create(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📅  APPOINTMENT  ➜  CREATE")
    print("-" * 60)
    print(f"  Provider : {args.get('provider_name', '<not specified>')}")
    print(f"  Patient  : {args.get('patient_name', '<self>')}")
    print(f"  Start    : {args.get('start_time', '<not specified>')}")
    print(f"  End      : {args.get('end_time', '<not specified>')}")
    print(f"  Symptoms : {args.get('symptoms', '<none reported>')}")
    print("  ▸ Booking appointment …")
    print("=" * 60 + "\n")


def mock_appointment_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📅  APPOINTMENT  ➜  READ")
    print("-" * 60)
    print(f"  Timeframe : {args.get('timeframe', '<all>')}")
    print(f"  Provider  : {args.get('provider_name', '<any>')}")
    print("  ▸ Loading appointments …")
    print("=" * 60 + "\n")


def mock_appointment_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📅  APPOINTMENT  ➜  UPDATE")
    print("-" * 60)
    for k, v in args.items():
        print(f"  {k:12s} = {v!r}")
    print("  ▸ Updating appointment …")
    print("=" * 60 + "\n")


def mock_appointment_delete(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📅  APPOINTMENT  ➜  CANCEL")
    print("-" * 60)
    print(f"  ID : {args.get('target_id', '<unknown>')}")
    print("  ▸ Cancelling appointment …")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Agent 2 — Health Recording (Nurse)
# ═══════════════════════════════════════════════════════════════════════════

def mock_take_test(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🩺  TEST  ➜  TAKE TEST")
    print("-" * 60)
    print(f"  Type : {args.get('test_type', '<not specified>')}")
    print("  ▸ Preparing test …")
    print("  ▸ Please place the device and hold still.")
    print("=" * 60 + "\n")


def mock_vital_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📊  VITALS  ➜  READ HISTORY")
    print("-" * 60)
    print(f"  Vital Type : {args.get('vital_type', '<all>')}")
    print(f"  Timeframe  : {args.get('timeframe', '<all time>')}")
    print("  ▸ Loading vital history …")
    print("=" * 60 + "\n")


def mock_media_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🎬  MEDIA  ➜  READ")
    print("-" * 60)
    print(f"  Type      : {args.get('media_type', '<all>')}")
    print(f"  Timeframe : {args.get('timeframe', '<all time>')}")
    print("  ▸ Loading media files …")
    print("=" * 60 + "\n")


def mock_media_delete(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🎬  MEDIA  ➜  DELETE")
    print("-" * 60)
    print(f"  Media ID : {args.get('media_id', '<unknown>')}")
    print("  ▸ Deleting media …")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Agent 3 — Health Consultation (Doctor)
# ═══════════════════════════════════════════════════════════════════════════

def mock_message_send(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("💬  MESSAGE  ➜  SEND")
    print("-" * 60)
    print(f"  Content : {args.get('content', '<empty>')}")
    print("  ▸ Sending message to doctor …")
    print("=" * 60 + "\n")


def mock_message_attach(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📎  MESSAGE  ➜  ATTACH FILE")
    print("-" * 60)
    print(f"  File Type : {args.get('file_type', '<any>')}")
    print("  ▸ Attaching file …")
    print("=" * 60 + "\n")


def mock_routine_create(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📋  ROUTINE  ➜  CREATE")
    print("-" * 60)
    print(f"  Name      : {args.get('name', '<not specified>')}")
    print(f"  Type      : {args.get('type', '<not specified>')}")
    print(f"  Frequency : {args.get('frequency', '<not specified>')}")
    print(f"  Time      : {args.get('time', '<not specified>')}")
    print(f"  Duration  : {args.get('duration', '<not specified>')}")
    print("  ▸ Creating routine …")
    print("=" * 60 + "\n")


def mock_routine_read(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📋  ROUTINE  ➜  READ")
    print("-" * 60)
    print(f"  Category  : {args.get('category', '<all>')}")
    print(f"  Timeframe : {args.get('timeframe', '<all>')}")
    print("  ▸ Loading routines …")
    print("=" * 60 + "\n")


def mock_routine_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📋  ROUTINE  ➜  UPDATE")
    print("-" * 60)
    for k, v in args.items():
        print(f"  {k:12s} = {v!r}")
    print("  ▸ Updating routine …")
    print("=" * 60 + "\n")


def mock_routine_delete(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📋  ROUTINE  ➜  DELETE")
    print("-" * 60)
    print(f"  Routine ID : {args.get('routine_id', '<unknown>')}")
    print("  ▸ Deleting routine …")
    print("=" * 60 + "\n")


def mock_meal_times_update(args: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🍽️  MEAL TIMES  ➜  UPDATE")
    print("-" * 60)
    print(f"  Meal : {args.get('meal', '<not specified>')}")
    print(f"  Time : {args.get('time', '<not specified>')}")
    print("  ▸ Updating meal time …")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Fallback
# ═══════════════════════════════════════════════════════════════════════════

def mock_fallback_prompt() -> None:
    print("\n" + "=" * 60)
    print("⚠️   FALLBACK — Intent not recognised")
    print("-" * 60)
    print("  \"I didn't quite catch that.")
    print("   You can ask me to manage your profile, book appointments,")
    print("   take health tests, view vitals, manage routines,")
    print("   or adjust settings.\"")
    print("=" * 60 + "\n")


def prompt_missing_fields(result: PipelineResult) -> None:
    """Display which fields are filled and which still need user input."""
    print("\n" + "=" * 60)
    print(f"❓  NEED MORE INFO  ➜  {result.action}")
    print("-" * 60)
    print(f"  Agent  : {result.agent}")
    print(f"  Action : {result.action}")
    if result.filled_args:
        print("  Already have:")
        for k, v in result.filled_args.items():
            print(f"    ✓ {k:20s} = {v!r}")
    print("  Still need:")
    for f in result.missing_fields:
        pretty = f.replace("_", " ").title()
        print(f"    ✗ {pretty}")
    print("-" * 60)
    print("  \"Could you please provide the missing information?\"")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Dispatch table
# ═══════════════════════════════════════════════════════════════════════════

TOOL_DISPATCH: Dict[str, Callable] = {
    # ── Agent 1: Receptionist ─────────────────────────────────────────
    "profile.read": mock_profile_read,
    "profile.update": mock_profile_update,
    "camera.open": mock_camera_open,
    "profile.verifyPhone": mock_verify_phone,
    "profile.verifyEmail": mock_verify_email,
    "diagnosis.read": mock_diagnosis_read,
    "allergies.create": mock_allergies_create,
    "allergies.read": mock_allergies_read,
    "allergies.update": mock_allergies_update,
    "allergies.delete": mock_allergies_delete,
    "careTeam.read": mock_careteam_read,
    "family.create": mock_family_create,
    "family.read": mock_family_read,
    "family.update": mock_family_update,
    "family.delete": mock_family_delete,
    "family.permissionsSchema": mock_family_permissions,
    "wifi.read": mock_wifi_read,
    "brightness.update": mock_brightness_update,
    "textSize.update": mock_textsize_update,
    "device.connect": mock_device_connect,
    "device.read": mock_device_read,
    "device.delete": mock_device_delete,
    "notification.read": mock_notification_read,
    "notification.update": mock_notification_update,
    "smartPrompt.update": mock_smart_prompt_update,
    "callSettings.read": mock_call_settings_read,
    "callSettings.update": mock_call_settings_update,
    "appointment.create": mock_appointment_create,
    "appointment.read": mock_appointment_read,
    "appointment.update": mock_appointment_update,
    "appointment.delete": mock_appointment_delete,
    # ── Agent 2: Nurse ────────────────────────────────────────────────
    "takeTest": mock_take_test,
    "vital.read": mock_vital_read,
    "media.read": mock_media_read,
    "media.delete": mock_media_delete,
    # ── Agent 3: Doctor ───────────────────────────────────────────────
    "message.send": mock_message_send,
    "message.attach": mock_message_attach,
    "routine.create": mock_routine_create,
    "routine.read": mock_routine_read,
    "routine.update": mock_routine_update,
    "routine.delete": mock_routine_delete,
    "mealTimes.update": mock_meal_times_update,
}


# ═══════════════════════════════════════════════════════════════════════════
# Route function
# ═══════════════════════════════════════════════════════════════════════════

def route(result: PipelineResult) -> str:
    """Route a PipelineResult to the appropriate handler.

    Returns:
        'ok'       — handler executed successfully
        'missing'  — required fields are missing (caller should prompt user)
        'fallback' — no agent/action classified
    """
    agent = result.agent
    action = result.action

    logger.info("Routing  agent=%r  action=%r", agent, action)

    if agent is None or action is None:
        logger.warning("No agent/action classified — triggering fallback.")
        mock_fallback_prompt()
        return "fallback"

    # Check for missing required fields
    if result.missing_fields:
        prompt_missing_fields(result)
        return "missing"

    handler = TOOL_DISPATCH.get(action)
    if handler:
        handler(result.filled_args)
        return "ok"
    else:
        logger.warning("No handler for action %r — fallback.", action)
        mock_fallback_prompt()
        return "fallback"
