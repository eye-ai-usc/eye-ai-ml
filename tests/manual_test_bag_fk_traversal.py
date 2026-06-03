"""
Regression tests to verify that downloading dataset bags correctly
follows FK paths to include all reachable records.

Bug 1: When a dataset's members are Images (linked via Image_Dataset),
the bag export does not follow the FK path Image.Observation -> Observation -> Subject,
leaving Observation and Subject tables empty in the downloaded bag even though
Image.Observation FK values are populated.
Affected dataset: 4-SS8W (version 0.3.0) on www.eye-ai.org

Bug 2: When a dataset's members include Observations, the bag export does not
follow the FK path Observation -> Report_HVF -> OCR_HVF, leaving OCR_HVF
empty in the downloaded bag even though Report_HVF.Observation FK values are populated.
Affected dataset: 2-C9PR (version 2.10.0) on www.eye-ai.org
"""
import logging
import pandas as pd
from eye_ai.eye_ai import EyeAI
from deriva_ml.dataset.aux_classes import DatasetSpec

ai = EyeAI(
    hostname="www.eye-ai.org",
    catalog_id="eye-ai",
    cache_dir="/tmp/eye_ai_cache",
    working_dir="/tmp/eye_ai_work",
    logging_level=logging.ERROR,
    deriva_logging_level=logging.ERROR,
)


# ---------------------------------------------------------------------------
# Test 1: Dataset 4-SS8W — FK path Image → Observation → Subject
# ---------------------------------------------------------------------------
DATASET_1_RID = "4-SS8W"
DATASET_1_VERSION = "0.3.0"

print(f"\n{'='*60}")
print(f"Test 1: Dataset {DATASET_1_RID} v{DATASET_1_VERSION}")
print(f"        FK path: Image → Observation → Subject")
print(f"{'='*60}")
print(f"Downloading dataset {DATASET_1_RID} v{DATASET_1_VERSION}...")

ds_bag_1 = ai.download_dataset_bag(DatasetSpec(rid=DATASET_1_RID, version=DATASET_1_VERSION))

image_rows        = pd.DataFrame(list(ds_bag_1.get_table_as_dict('Image')))
obs_rows          = pd.DataFrame(list(ds_bag_1.get_table_as_dict('Observation')))
subject_rows      = pd.DataFrame(list(ds_bag_1.get_table_as_dict('Subject')))
image_dataset_rows = pd.DataFrame(list(ds_bag_1.get_table_as_dict('Image_Dataset')))

print(f"\nImage_Dataset (association table): {len(image_dataset_rows)} rows")
print(f"Image:                             {len(image_rows)} rows")
print(f"Observation:                       {len(obs_rows)} rows  <-- expected > 0")
print(f"Subject:                           {len(subject_rows)} rows  <-- expected > 0")

if not image_rows.empty:
    print(f"\nImage.Observation FK values (should be non-null):")
    print(image_rows['Observation'].value_counts(dropna=False))

has_images   = len(image_rows) > 0
has_obs_fk   = not image_rows.empty and image_rows['Observation'].notna().any()
obs_missing  = len(obs_rows) == 0
subj_missing = len(subject_rows) == 0

print("\n--- Bug Verification ---")
print(f"Images present in bag:              {has_images}")
print(f"Image.Observation FK is populated:  {has_obs_fk}")
print(f"Observation missing from bag:       {obs_missing}  <-- True = bug confirmed")
print(f"Subject missing from bag:           {subj_missing}  <-- True = bug confirmed")

if has_images and has_obs_fk and obs_missing and subj_missing:
    print("\nBUG CONFIRMED: FK traversal Image -> Observation -> Subject was not followed during bag export.")
else:
    print("\nBug not reproduced.")


# ---------------------------------------------------------------------------
# Test 2: Dataset 2-C9PR — FK path Observation → Report_HVF → OCR_HVF
# ---------------------------------------------------------------------------
DATASET_2_RID = "2-C9PR"
DATASET_2_VERSION = "2.10.0"

print(f"\n{'='*60}")
print(f"Test 2: Dataset {DATASET_2_RID} v{DATASET_2_VERSION}")
print(f"        FK path: Observation → Report_HVF → OCR_HVF")
print(f"{'='*60}")
print(f"Downloading dataset {DATASET_2_RID} v{DATASET_2_VERSION}...")

ds_bag_2 = ai.download_dataset_bag(DatasetSpec(rid=DATASET_2_RID, version=DATASET_2_VERSION))

obs_rows_2       = pd.DataFrame(list(ds_bag_2.get_table_as_dict('Observation')))
report_hvf_rows  = pd.DataFrame(list(ds_bag_2.get_table_as_dict('Report_HVF')))
ocr_hvf_rows     = pd.DataFrame(list(ds_bag_2.get_table_as_dict('OCR_HVF')))

print(f"\nObservation:  {len(obs_rows_2)} rows")
print(f"Report_HVF:   {len(report_hvf_rows)} rows  <-- expected > 0")
print(f"OCR_HVF:      {len(ocr_hvf_rows)} rows  <-- expected > 0")

if not report_hvf_rows.empty and 'Observation' in report_hvf_rows.columns:
    print(f"\nReport_HVF.Observation FK values (should be non-null):")
    print(report_hvf_rows['Observation'].value_counts(dropna=False))

has_obs        = len(obs_rows_2) > 0
has_report_hvf = len(report_hvf_rows) > 0
has_obs_fk_2   = not report_hvf_rows.empty and 'Observation' in report_hvf_rows.columns and report_hvf_rows['Observation'].notna().any()
ocr_missing    = len(ocr_hvf_rows) == 0

print("\n--- Bug Verification ---")
print(f"Observations present in bag:            {has_obs}")
print(f"Report_HVF present in bag:              {has_report_hvf}")
print(f"Report_HVF.Observation FK is populated: {has_obs_fk_2}")
print(f"OCR_HVF missing from bag:               {ocr_missing}  <-- True = bug confirmed")

if has_report_hvf and has_obs_fk_2 and ocr_missing:
    print("\nBUG CONFIRMED: FK traversal Observation -> Report_HVF -> OCR_HVF was not followed during bag export.")
else:
    print("\nBug not reproduced.")
