"""Read-only integration test for EyeAI against a live catalog.

Exercises ``EyeAI.image_tall`` end-to-end against the ``eye-ai`` catalog on
``dev.eye-ai.org`` (which already carries the full eye-ai schema and data, so
no ephemeral test-catalog harness is needed). ``image_tall`` is chosen because
it exercises the restored ``DerivaML.user_list()`` path (mapping a grader's
RCB user id to Full_Name) in addition to bag denormalization.

This test downloads a dataset bag and reads from it; it makes no catalog
mutations. It skips cleanly when no test host / credentials are configured, so
it is safe in environments without dev access.

Environment variables:
  EYE_AI_TEST_HOSTNAME   catalog host (default: dev.eye-ai.org)
  EYE_AI_TEST_CATALOG    catalog id   (default: eye-ai)
  EYE_AI_TEST_DATASET    dataset RID to read
  EYE_AI_TEST_VERSION    dataset version
  EYE_AI_TEST_DIAG_TAG   diagnosis tag to filter on (default: Initial Diagnosis)

Run against dev explicitly, e.g.:
  EYE_AI_TEST_DATASET=2-C9PR EYE_AI_TEST_VERSION=2.10.0 \
    uv run pytest tests/test_eye_ai.py -v
"""

import os

import pytest

from deriva_ml.dataset.aux_classes import DatasetSpec

DATASET_RID = os.getenv("EYE_AI_TEST_DATASET")
DATASET_VERSION = os.getenv("EYE_AI_TEST_VERSION")
DIAG_TAG = os.getenv("EYE_AI_TEST_DIAG_TAG", "Initial Diagnosis")

IMAGE_TALL_COLUMNS = {
    "Subject_RID",
    "Image_RID",
    "Diagnosis_RID",
    "Full_Name",
    "Image_Side",
    "Diagnosis_Image",
    "Cup_Disk_Ratio",
    "Image_Quality",
}

# A dataset RID/version must be supplied to run the live image_tall test.
needs_dataset = pytest.mark.skipif(
    not (DATASET_RID and DATASET_VERSION),
    reason="Set EYE_AI_TEST_DATASET and EYE_AI_TEST_VERSION to run the live image_tall test.",
)


class TestUserList:
    """user_list() is the restored deriva-ml accessor image_tall depends on."""

    def test_user_list_returns_id_and_full_name(self, eye_ai):
        users = eye_ai.user_list()
        assert isinstance(users, list)
        assert len(users) >= 1
        for user in users:
            assert "ID" in user and "Full_Name" in user


@needs_dataset
class TestImageTall:
    def test_image_tall_shape_and_columns(self, eye_ai):
        ds_bag = eye_ai.download_dataset_bag(
            DatasetSpec(rid=DATASET_RID, version=DATASET_VERSION, materialize=False)
        )
        result = eye_ai.image_tall(ds_bag, DIAG_TAG)

        # Contract: the documented column set, with at least one row.
        assert set(result.columns) == IMAGE_TALL_COLUMNS
        assert len(result) > 0
        # Every row carries a Full_Name (resolved via user_list for grading tags,
        # or assigned the diagnosis_tag otherwise).
        assert result["Full_Name"].notna().all()
