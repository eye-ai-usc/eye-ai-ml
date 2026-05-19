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

print(f"Downloading dataset {args.dataset_rid}...")
ds_bag = ai.download_dataset_bag(DatasetSpec(rid=args.dataset_rid, version=args.dataset_version, materialize=False))

print("Running severity_analysis...")
result = ai.severity_analysis(ds_bag)

print(f"\nResult shape: {result.shape}")

# Sanity checks
assert result.shape[0] > 0, "Result should have rows"
assert "RNFL_severe" in result.columns, "Missing RNFL_severe"
assert "HVF_severe" in result.columns, "Missing HVF_severe"
assert "CDR_severe" in result.columns, "Missing CDR_severe"
assert "Severity_Mismatch" in result.columns, "Missing Severity_Mismatch"

n_subjects = result['Subject.RID'].nunique()
n_mismatch = result['Severity_Mismatch'].sum()
n_rows = result.shape[0]

print(f"\nSubjects: {n_subjects}")
print(f"Rows (subjects x sides): {n_rows}")

print("\nSeverity columns (sample of 10 rows):")
print(result[['Subject.RID', 'Image_Side', 'RNFL_severe', 'HVF_severe', 'CDR_severe', 'Severity_Mismatch']].head(10).to_string())

print(f"\nSeverity_Mismatch: {n_mismatch} / {n_rows} rows ({100*n_mismatch/n_rows:.1f}%)")

print("\nRNFL_severe distribution:")
print(result['RNFL_severe'].value_counts(dropna=False).to_string())

print("\nHVF_severe distribution:")
print(result['HVF_severe'].value_counts(dropna=False).to_string())

print("\nCDR_severe distribution:")
print(result['CDR_severe'].value_counts(dropna=False).to_string())

print("\nAll checks passed.")
