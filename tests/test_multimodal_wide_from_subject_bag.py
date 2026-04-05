import argparse
import logging
import pandas as pd
from eye_ai.eye_ai import EyeAI
from deriva_ml.dataset.aux_classes import DatasetSpec

parser = argparse.ArgumentParser()
parser.add_argument("--hostname", default="dev.eye-ai.org", help="Catalog hostname")
parser.add_argument("--image-dataset-rid", default="5-STDA", help="Image dataset RID")
parser.add_argument("--image-dataset-version", default="3.6.0", help="Image dataset version")
parser.add_argument("--subject-dataset-rid", default="2-C9PP", help="Subject dataset RID")
parser.add_argument("--subject-dataset-version", default="3.6.0", help="Subject dataset version")
args = parser.parse_args()

ai = EyeAI(
    hostname=args.hostname,
    catalog_id="eye-ai",
    cache_dir="/tmp/eye_ai_cache",
    working_dir="/tmp/eye_ai_work",
    logging_level=logging.ERROR,
    deriva_logging_level=logging.ERROR,
)

print(f"Downloading subject dataset {args.subject_dataset_rid} v{args.subject_dataset_version}...")
subject_bag = ai.download_dataset_bag(DatasetSpec(
    rid=args.subject_dataset_rid,
    version=args.subject_dataset_version,
    materialize=False,
))
print("Subject dataset downloaded.")

image_spec = DatasetSpec(
    rid=args.image_dataset_rid,
    version=args.image_dataset_version,
    materialize=False,
)
print(f"\nRunning multimodal_wide_from_subject_bag on {image_spec.rid} v{image_spec.version}...")
result = ai.multimodal_wide_from_subject_bag(image_spec, subject_bag)

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

print("\nCoverage:")
n_subjects = result['Subject.RID'].nunique()
print(f"  Subjects: {n_subjects}")
print(f"  Total rows: {result.shape[0]} ({n_subjects} subjects x 2 sides)")
print(f"  Clinical Records: {result['Clinical_Records.RID'].notna().sum()} / {result.shape[0]}")
print(f"  HVF: {result['Report_HVF.RID'].notna().sum()} / {result.shape[0]}")
print(f"  RNFL: {result['Report_RNFL.RID'].notna().sum()} / {result.shape[0]}")

print("\nAll checks passed.")
