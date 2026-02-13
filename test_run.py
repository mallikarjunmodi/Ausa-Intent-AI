#!/usr/bin/env python3
"""
test_run.py — Automated test scenarios for the hierarchical agent pipeline.

Runs in text-only mode (no ASR needed). Tests all 3 agents and key tools.
"""

import logging
import sys
from typing import List, Tuple

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

# ═══════════════════════════════════════════════════════════════════════════
# Test definitions: (description, input_text, expected_agent, expected_action)
# ═══════════════════════════════════════════════════════════════════════════

TEST_CASES: List[Tuple[str, str, str, str]] = [
    # ── Agent 1: Receptionist ─────────────────────────────────────────
    (
        "View profile",
        "show me my profile",
        "receptionist",
        "profile.read",
    ),
    (
        "Update profile name",
        "change my name to Rahul",
        "receptionist",
        "profile.update",
    ),
    (
        "Add allergy",
        "add an allergy to peanuts, severity is high",
        "receptionist",
        "allergies.create",
    ),
    (
        "View allergies",
        "show my allergies",
        "receptionist",
        "allergies.read",
    ),
    (
        "View care team",
        "who is on my care team",
        "receptionist",
        "careTeam.read",
    ),
    (
        "Add family member",
        "invite my wife to family, email is wife@test.com",
        "receptionist",
        "family.create",
    ),
    (
        "Create appointment with symptoms",
        "I'm Shreya and I dont like to smile, create a consultation with doctor",
        "receptionist",
        "appointment.create",
    ),
    (
        "View appointments",
        "show my appointments this week",
        "receptionist",
        "appointment.read",
    ),
    (
        "Change brightness",
        "set brightness to 80%",
        "receptionist",
        "brightness.update",
    ),
    (
        "View connected devices",
        "show connected devices",
        "receptionist",
        "device.read",
    ),

    # ── Agent 2: Nurse ────────────────────────────────────────────────
    (
        "Take blood pressure test",
        "take my blood pressure",
        "nurse",
        "takeTest",
    ),
    (
        "Take ECG test",
        "I want to do an ECG test",
        "nurse",
        "takeTest",
    ),
    (
        "View vital history",
        "what was my last blood pressure reading",
        "nurse",
        "vital.read",
    ),
    (
        "View blood glucose history",
        "show my blood glucose history for past week",
        "nurse",
        "vital.read",
    ),

    # ── Agent 3: Doctor ───────────────────────────────────────────────
    (
        "Create routine",
        "create a blood pressure routine daily at 8 AM",
        "doctor",
        "routine.create",
    ),
    (
        "View routines",
        "show all my routines",
        "doctor",
        "routine.read",
    ),
    (
        "Update meal times",
        "set breakfast time to 8 AM",
        "doctor",
        "mealTimes.update",
    ),
    (
        "Send message",
        "send a message to my doctor",
        "doctor",
        "message.send",
    ),

    # ── Edge cases / Fallback ─────────────────────────────────────────
    (
        "Feeling sick appointment (symptom keywords)",
        "Im rahul and im feeling sick, create an appointment",
        "receptionist",
        "appointment.create",
    ),
    (
        "Ambiguous query (should not crash)",
        "hello how are you",
        None,
        None,
    ),
]


def run_tests():
    from src.nlu.extractor import IntentExtractor
    from src.router.handler import route

    print("\n" + "=" * 70)
    print("  AUSA HEALTH — Pipeline Test Suite")
    print("=" * 70)
    print(f"  {len(TEST_CASES)} test cases\n")

    print("  ⏳  Loading GLiNER model …")
    nlu = IntentExtractor()
    nlu._load_model()
    print("  ✅  Ready!\n")

    passed = 0
    failed = 0

    for i, (desc, text, exp_agent, exp_action) in enumerate(TEST_CASES, 1):
        result = nlu.analyse(text)

        agent_ok = result.agent == exp_agent
        action_ok = result.action == exp_action
        ok = agent_ok and action_ok

        icon = "✅" if ok else "❌"
        print(f"  {icon}  [{i:02d}] {desc}")
        print(f"        Input  : \"{text}\"")
        print(f"        Agent  : {result.agent!r:18s} {'✓' if agent_ok else '✗ expected ' + repr(exp_agent)}")
        print(f"        Action : {result.action!r:18s} {'✓' if action_ok else '✗ expected ' + repr(exp_action)}")

        if result.filled_args:
            print(f"        Filled : {result.filled_args}")
        if result.missing_fields:
            print(f"        Missing: {result.missing_fields}")
        print()

        if ok:
            passed += 1
        else:
            failed += 1

        # Also route (to verify handlers don't crash)
        route(result)

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 70)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  🎉  All tests passed!")
    else:
        print("  ⚠️   Some tests failed — review output above.")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
