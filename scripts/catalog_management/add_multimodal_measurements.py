"""
End-to-end script to add matched HVF, RNFL, and Clinical Records from a subject
dataset into an image dataset as members.

This is a one-time (or occasional) catalog modification. It:
  1. Downloads the subject dataset bag.
  2. Calls add_multimodal_measurements, which downloads the image bag, runs the
     selection logic, registers the necessary element types, and adds members.
  3. Prints the number of members added per table.

The image dataset minor version is bumped automatically by add_dataset_members.

Test against dev first:
    uv run python scripts/catalog_management/add_multimodal_measurements.py

Then run against prod:
    uv run python scripts/catalog_management/add_multimodal_measurements.py \\
        --hostname www.eye-ai.org
"""

import argparse
import logging

from deriva_ml.dataset.aux_classes import DatasetSpec
from eye_ai.eye_ai import EyeAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add matched HVF/RNFL/Clinical Records to an image dataset."
    )
    parser.add_argument("--hostname", default="dev.eye-ai.org",
                        help="Catalog hostname (default: dev.eye-ai.org)")
    parser.add_argument("--catalog-id", default="eye-ai",
                        help="Catalog ID (default: eye-ai)")
    parser.add_argument("--image-dataset-rid", default="4-411G",
                        help="RID of the image dataset to enrich (default: 4-411G)")
    parser.add_argument("--image-dataset-version", default="2.10.0",
                        help="Version of the image dataset (default: 2.10.0)")
    parser.add_argument("--subject-dataset-rid", default="2-C9PP",
                        help="RID of the subject dataset to draw measurements from (default: 2-C9PP)")
    parser.add_argument("--subject-dataset-version", default="3.6.0",
                        help="Version of the subject dataset (default: 3.6.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ai = EyeAI(
        hostname=args.hostname,
        catalog_id=args.catalog_id,
        cache_dir="/tmp/eye_ai_cache",
        working_dir="/tmp/eye_ai_work",
        logging_level=logging.INFO,
        deriva_logging_level=logging.ERROR,
    )

    print(f"Connected to {args.hostname} / {args.catalog_id}")

    subject_spec = DatasetSpec(
        rid=args.subject_dataset_rid,
        version=args.subject_dataset_version,
        materialize=False,
    )
    print(f"Downloading subject dataset {subject_spec.rid} v{subject_spec.version}...")
    subject_bag = ai.download_dataset_bag(subject_spec)
    print("Subject dataset downloaded.")

    image_spec = DatasetSpec(
        rid=args.image_dataset_rid,
        version=args.image_dataset_version,
        materialize=False,
    )
    print(f"\nRunning add_multimodal_measurements on image dataset {image_spec.rid} v{image_spec.version}...")
    counts = ai.add_multimodal_measurements(
        image_dataset=image_spec,
        subject_bag=subject_bag,
    )

    if counts:
        print("\nMembers added:")
        for table, n in counts.items():
            print(f"  {table}: {n}")
    else:
        print("\nNo members added (no matching records found).")

    updated_dataset = ai.lookup_dataset(image_spec.rid)
    print(f"\nImage dataset {image_spec.rid} is now at version {updated_dataset.current_version}")


if __name__ == "__main__":
    main()
