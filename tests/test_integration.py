"""Read-only integration tests for EyeAI's bag-transform analytics.

Covers extract_modality, multimodal_wide, severity_analysis, and
multimodal_wide_from_subject_bag against a live catalog. These mirror the
former argparse runner scripts (kept as ``manual_*.py`` for verbose manual
runs) but assert the contract instead of printing.

All tests use the session-scoped ``eye_ai`` fixture (see conftest.py) and skip
cleanly without catalog access. Dataset RIDs default to known-good values and
are overridable via env vars.
"""

import os

import pytest

from deriva_ml.dataset.aux_classes import DatasetSpec

# Single-bag multimodal dataset (HVF/RNFL/Clinical reachable from one bag).
WIDE_RID = os.getenv("EYE_AI_TEST_WIDE_DATASET", "2-C9PR")
WIDE_VERSION = os.getenv("EYE_AI_TEST_WIDE_VERSION", "2.10.0")

# Image-bag + subject-bag pair for the two-bag workaround path.
IMAGE_RID = os.getenv("EYE_AI_TEST_IMAGE_DATASET", "5-STDA")
IMAGE_VERSION = os.getenv("EYE_AI_TEST_IMAGE_VERSION", "3.6.0")
SUBJECT_RID = os.getenv("EYE_AI_TEST_SUBJECT_DATASET", "2-C9PP")
SUBJECT_VERSION = os.getenv("EYE_AI_TEST_SUBJECT_VERSION", "3.6.0")


@pytest.fixture(scope="module")
def wide_bag(eye_ai):
    return eye_ai.download_dataset_bag(
        DatasetSpec(rid=WIDE_RID, version=WIDE_VERSION, materialize=False)
    )


class TestExtractModality:
    def test_returns_four_modalities(self, eye_ai, wide_bag):
        modality = eye_ai.extract_modality(wide_bag)
        assert set(modality.keys()) == {"Clinic", "HVF", "RNFL", "Fundus"}
        assert modality["Fundus"].shape[0] > 0


class TestMultimodalWide:
    def test_wide_table_shape_and_no_merge_suffixes(self, eye_ai, wide_bag):
        result = eye_ai.multimodal_wide(wide_bag)
        assert result.shape[0] > 0
        assert "Subject.RID" in result.columns
        assert "Image_Side" in result.columns
        # Clean joins: no pandas _x / _y merge-collision suffixes.
        suffixed = [c for c in result.columns if c.endswith("_x") or c.endswith("_y")]
        assert not suffixed, f"merge-collision columns: {suffixed}"


class TestSeverityAnalysis:
    def test_severity_columns_present(self, eye_ai, wide_bag):
        result = eye_ai.severity_analysis(wide_bag)
        assert result.shape[0] > 0
        for col in ("RNFL_severe", "HVF_severe", "CDR_severe", "Severity_Mismatch"):
            assert col in result.columns


class TestMultimodalWideFromSubjectBag:
    def test_two_bag_join(self, eye_ai):
        subject_bag = eye_ai.download_dataset_bag(
            DatasetSpec(rid=SUBJECT_RID, version=SUBJECT_VERSION, materialize=False)
        )
        image_bag = eye_ai.download_dataset_bag(
            DatasetSpec(rid=IMAGE_RID, version=IMAGE_VERSION, materialize=False)
        )
        result = eye_ai.multimodal_wide_from_subject_bag(image_bag, subject_bag)
        assert result.shape[0] > 0
        assert "Subject.RID" in result.columns
        assert "Image_Side" in result.columns
