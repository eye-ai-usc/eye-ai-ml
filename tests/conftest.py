"""Shared fixtures for EyeAI tests.

Integration tests connect to a live catalog (default ``dev.eye-ai.org``).
Because ``EyeAI()`` blocks for a long time on an unreachable host rather than
failing fast, the live suite is **opt-in**: it is skipped unless
``EYE_AI_RUN_INTEGRATION=1`` is set. This keeps the default ``pytest`` run
(unit tests only) fast and hang-free in CI / dev machines without catalog
access. Unit tests (``test_eye_ai_units``) use no fixtures from here.

Enable the live suite with:
  EYE_AI_RUN_INTEGRATION=1 EYE_AI_TEST_DATASET=2-C9PR EYE_AI_TEST_VERSION=2.10.0 \
    uv run pytest tests/
"""

import logging
import os

import pytest

from eye_ai.eye_ai import EyeAI

HOSTNAME = os.getenv("EYE_AI_TEST_HOSTNAME", "dev.eye-ai.org")
CATALOG_ID = os.getenv("EYE_AI_TEST_CATALOG", "eye-ai")
RUN_INTEGRATION = os.getenv("EYE_AI_RUN_INTEGRATION") == "1"


@pytest.fixture(scope="session")
def eye_ai():
    """Connect to the eye-ai catalog.

    Skips unless EYE_AI_RUN_INTEGRATION=1 (live tests are opt-in because the
    connection does not fail fast on an unreachable host). Also skips on any
    connection/auth error.
    """
    if not RUN_INTEGRATION:
        pytest.skip("Live integration tests are opt-in; set EYE_AI_RUN_INTEGRATION=1 to run.")
    try:
        ai = EyeAI(
            hostname=HOSTNAME,
            catalog_id=CATALOG_ID,
            cache_dir="/tmp/eye_ai_cache",
            working_dir="/tmp/eye_ai_work",
            logging_level=logging.ERROR,
            deriva_logging_level=logging.ERROR,
        )
    except Exception as exc:  # noqa: BLE001 - any connection/auth failure -> skip
        pytest.skip(f"Cannot connect to {HOSTNAME}/{CATALOG_ID}: {exc}")
    return ai
