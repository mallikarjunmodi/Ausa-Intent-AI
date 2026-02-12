"""
handler.py — Intent Router & Mock Action Handlers

Receives the ``AnalysisResult`` from the NLU node and dispatches
to the correct mock handler via ``if / elif / else`` logic.

Routing rules:
  ┌───────────────────────┬─────────────────────────────┐
  │ Detected intent       │ Action                      │
  ├───────────────────────┼─────────────────────────────┤
  │ view_result           │ mock_view_result()           │
  │ create_routine        │ mock_create_routine()        │
  │ <missing / unknown>   │ mock_fallback_prompt()       │
  └───────────────────────┴─────────────────────────────┘

Usage:
    from src.router.handler import route
    from src.nlu.extractor import AnalysisResult

    route(analysis_result)
"""

from __future__ import annotations

import logging
from typing import Optional

from src.nlu.extractor import AnalysisResult

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Mock Action Functions
# ═══════════════════════════════════════════════════════════════════════════

def mock_view_result(
    target: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> None:
    """Simulate retrieving and displaying a health reading.

    In production this would query the on-device health database and
    render results on the QCS6490's display.

    Args:
        target:    The sensor / metric to look up (e.g. "blood pressure").
        timeframe: Human-readable time window   (e.g. "last week").
    """
    print("\n" + "=" * 60)
    print("✅  ACTION  ➜  VIEW RESULT")
    print("-" * 60)
    print(f"  Sensor / Target : {target or '<not specified>'}")
    print(f"  Timeframe       : {timeframe or '<not specified>'}")
    print("-" * 60)
    print("  ▸ Fetching stored readings from local health DB …")
    print("  ▸ Rendering result card on-screen.")
    print("=" * 60 + "\n")


def mock_create_routine(
    target: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> None:
    """Simulate scheduling a new health-monitoring routine.

    In production this would persist the routine config to the on-device
    scheduler and confirm via audio/visual feedback.

    Args:
        target:    The sensor / metric to schedule (e.g. "blood pressure").
        timeframe: When to run the routine           (e.g. "morning").
    """
    print("\n" + "=" * 60)
    print("✅  ACTION  ➜  CREATE ROUTINE")
    print("-" * 60)
    print(f"  Sensor / Target : {target or '<not specified>'}")
    print(f"  Timeframe       : {timeframe or '<not specified>'}")
    print("-" * 60)
    print("  ▸ Saving new routine to on-device scheduler …")
    print("  ▸ Confirmation sent to display.")
    print("=" * 60 + "\n")


def mock_fallback_prompt() -> None:
    """Prompt the user to rephrase when intent is unclear.

    Triggered when:
      • No intent could be classified from the input text.
      • The classified intent string does not match any known route.
    """
    print("\n" + "=" * 60)
    print("⚠️   FALLBACK — Intent not recognised")
    print("-" * 60)
    print("  \"I didn't quite catch that.")
    print("   Did you want to view a result or start a new reading?\"")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Route Dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def route(result: AnalysisResult) -> None:
    """Dispatch an NLU analysis result to the correct mock handler.

    Uses ``if / elif / else`` on the classified intent string.
    Falls back to ``mock_fallback_prompt()`` when no intent was
    classified or the intent is unrecognised.

    Args:
        result: The ``AnalysisResult`` produced by
                ``IntentExtractor.analyse()``.
    """
    intent: Optional[str] = result.intent
    target: Optional[str] = result.sensor_target
    timeframe: Optional[str] = result.timeframe

    logger.info(
        "Routing  intent=%r  target=%r  timeframe=%r",
        intent, target, timeframe,
    )

    # --- No intent classified → immediate fallback -------------------------
    if intent is None:
        logger.warning("No intent classified — triggering fallback.")
        mock_fallback_prompt()
        return

    # --- if / elif / else dispatch -----------------------------------------
    if intent == "view_result":
        mock_view_result(target=target, timeframe=timeframe)

    elif intent == "create_routine":
        mock_create_routine(target=target, timeframe=timeframe)

    # Anything else → fallback
    else:
        logger.warning("Unrecognised intent %r — triggering fallback.", intent)
        mock_fallback_prompt()
