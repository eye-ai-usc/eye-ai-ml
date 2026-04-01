"""
Regression test to verify that downloading an image dataset bag correctly
includes FK-reachable Observation and Subject records.

Bug report: When a dataset's members are Images (linked via Image_Dataset),
the bag export does not follow the FK path Image.Observation -> Observation -> Subject,
leaving Observation and Subject tables empty in the downloaded bag even though
Image.Observation FK values are populated.

Affected dataset: 4-SS8W (version 0.3.0) on www.eye-ai.org
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

DATASET_RID = "4-SS8W"
DATASET_VERSION = "0.3.0"

print(f"Downloading dataset {DATASET_RID} v{DATASET_VERSION}...")
ds_bag = ai.download_dataset_bag(DatasetSpec(rid=DATASET_RID, version=DATASET_VERSION))

image_rows = pd.DataFrame(list(ds_bag.get_table_as_dict('Image')))
obs_rows = pd.DataFrame(list(ds_bag.get_table_as_dict('Observation')))
subject_rows = pd.DataFrame(list(ds_bag.get_table_as_dict('Subject')))
image_dataset_rows = pd.DataFrame(list(ds_bag.get_table_as_dict('Image_Dataset')))

print(f"\nImage_Dataset (association table): {len(image_dataset_rows)} rows")
print(f"Image:                             {len(image_rows)} rows")
print(f"Observation:                       {len(obs_rows)} rows  <-- expected > 0")
print(f"Subject:                           {len(subject_rows)} rows  <-- expected > 0")

if not image_rows.empty:
    print(f"\nImage.Observation FK values (should be non-null):")
    print(image_rows['Observation'].value_counts(dropna=False))

# Verify the bug: Image has data and valid Observation FKs,
# but Observation and Subject are empty in the bag.
has_images = len(image_rows) > 0
has_obs_fk = not image_rows.empty and image_rows['Observation'].notna().any()
obs_missing = len(obs_rows) == 0
subject_missing = len(subject_rows) == 0

print("\n--- Bug Verification ---")
print(f"Images present in bag:              {has_images}")
print(f"Image.Observation FK is populated:  {has_obs_fk}")
print(f"Observation missing from bag:       {obs_missing}  <-- True = bug confirmed")
print(f"Subject missing from bag:           {subject_missing}  <-- True = bug confirmed")

if has_images and has_obs_fk and obs_missing and subject_missing:
    print("\nBUG CONFIRMED: FK traversal Image -> Observation -> Subject was not followed during bag export.")
else:
    print("\nBug not reproduced.")
