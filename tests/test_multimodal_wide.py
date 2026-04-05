import argparse
import logging
from eye_ai.eye_ai import EyeAI
from deriva_ml.dataset.aux_classes import DatasetSpec

parser = argparse.ArgumentParser()
parser.add_argument("--hostname", default="www.eye-ai.org", help="Catalog hostname")
parser.add_argument("--dataset-rid", default="2-C9PR", help="Dataset RID")
parser.add_argument("--dataset-version", default="2.10.0", help="Dataset version")
args = parser.parse_args()

ai = EyeAI(
    hostname=args.hostname,
    catalog_id="eye-ai",
    cache_dir="/tmp/eye_ai_cache",
    working_dir="/tmp/eye_ai_work",
    logging_level=logging.ERROR,
    deriva_logging_level=logging.ERROR,
)

DATASET_RID = args.dataset_rid
DATASET_VERSION = args.dataset_version

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

print("\nClinical Records coverage:")
n_subjects = result['Subject.RID'].nunique()
n_with_clinic = result['Clinical_Records.RID'].notna().sum()
print(f"  Subjects: {n_subjects}")
print(f"  Rows with Clinical Records: {n_with_clinic} / {result.shape[0]}")
print(f"  Rows without Clinical Records: {result['Clinical_Records.RID'].isna().sum()}")

print("\nHVF coverage:")
print(f"  Rows with HVF: {result['Report_HVF.RID'].notna().sum()} / {result.shape[0]}")

print("\nRNFL coverage:")
print(f"  Rows with RNFL: {result['Report_RNFL.RID'].notna().sum()} / {result.shape[0]}")

print("\nAll checks passed.")
