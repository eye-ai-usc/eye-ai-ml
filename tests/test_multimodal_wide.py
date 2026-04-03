import logging
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

# Replace with a multimodal dataset RID and version that contains
# Subject, Observation, Report_HVF, Report_RNFL, and Clinical_Records data.
DATASET_RID = "2-C9PR"
DATASET_VERSION = "2.10.0"

print(f"Downloading dataset {DATASET_RID}...")
ds_bag = ai.download_dataset_bag(DatasetSpec(rid=DATASET_RID, version=DATASET_VERSION, materialize=False))

print("\nDebug: Raw table row counts:")
import pandas as pd
for t in ['Subject', 'Observation', 'Report_HVF', 'OCR_HVF', 'Report_RNFL', 'OCR_RNFL', 'Clinical_Records']:
    rows = list(ds_bag.get_table_as_dict(t))
    print(f"  {t}: {len(rows)} rows")

print("\nDebug: HVF denormalized frame:")
hvf_frame = ds_bag.denormalize_as_dataframe(["Subject", "Observation", "Report_HVF", "OCR_HVF"])
print(f"  shape: {hvf_frame.shape}")
print(f"  Observation cols: {[c for c in hvf_frame.columns if c.startswith('Observation.')]}")

print("\nRunning extract_modality...")
modality_df = ai.extract_modality(ds_bag)
for name, df in modality_df.items():
    print(f"  {name}: {df.shape}")
    print(f"    columns: {list(df.columns)}")

print("\nRunning multimodal_wide...")
result = ai.multimodal_wide(ds_bag)
print(f"\nResult shape: {result.shape}")
print(f"Columns ({len(result.columns)}):")
for col in result.columns:
    print(f"  {col}")
print("\nFirst 3 rows:")
print(result.head(3).to_string())

# Basic sanity checks
assert result.shape[0] > 0, "Result should have rows"
assert "Subject.RID" in result.columns, "Missing Subject.RID"
assert "Image_Side" in result.columns, "Missing Image_Side"
assert not any("_x" in c or "_y" in c for c in result.columns), \
    f"Duplicate columns found: {[c for c in result.columns if '_x' in c or '_y' in c]}"
print("\nAll checks passed.")
