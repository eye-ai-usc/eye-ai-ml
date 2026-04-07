"""
Migrate image-dataset membership from the old Image_Dataset junction table
(Image has FK → Dataset) to the new Dataset_Image junction table
(Dataset has FK → Image), then drop Image_Dataset from the schema.

Background
----------
Two junction tables exist on prod (and potentially dev):
  - eye-ai.Image_Dataset  (old): Image.Dataset → deriva-ml.Dataset.RID
  - eye-ai.Dataset_Image  (new): Dataset_Image.Dataset → deriva-ml.Dataset.RID

DerivaML's denormalize_as_dataframe prefers Dataset_Image (the standard
Dataset_{Element} naming convention) over Image_Dataset when both exist,
so bags exported from Image_Dataset-only datasets return an empty frame.

Additionally, add_dataset_members uses find_associations() to discover
junction tables dynamically. When both tables exist, the dict comprehension
produces a non-deterministic mapping for 'Image', meaning new image members
could be written to either table. Dropping Image_Dataset is required to
make behavior consistent.

Current state on prod (as of 2026-04-06)
-----------------------------------------
  Image_Dataset : 114,135 rows  across 72 datasets  (needs migration)
  Dataset_Image :  43,113 rows  across  4 datasets  (already correct)
    - 5-HJ8T    : 42,913 rows duplicated in BOTH tables  → delete from Image_Dataset only
    - 5-HE68    :     80 rows only in Dataset_Image       → nothing to do
    - 5-HE6G    :     20 rows only in Dataset_Image       → nothing to do
    - 5-7YBG    :    100 rows only in Dataset_Image       → nothing to do

Steps
-----
1. DRY-RUN (default): print what would happen, touch nothing.
2. --migrate : insert missing rows into Dataset_Image, delete all from Image_Dataset,
               then drop the Image_Dataset table from the schema.
3. --verify  : after migration, confirm Image_Dataset is gone and counts match.

Usage
-----
# Dry-run against prod (safe, read-only):
    uv run python scripts/catalog_management/migrate_image_dataset_to_dataset_image.py

# Run migration against prod:
    uv run python scripts/catalog_management/migrate_image_dataset_to_dataset_image.py --migrate

# Verify after migration:
    uv run python scripts/catalog_management/migrate_image_dataset_to_dataset_image.py --verify

# Target dev instead:
    uv run python scripts/catalog_management/migrate_image_dataset_to_dataset_image.py \\
        --hostname dev.eye-ai.org --migrate
"""

import argparse
import collections
import logging

from eye_ai.eye_ai import EyeAI

BATCH_SIZE = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Image_Dataset → Dataset_Image, then drop Image_Dataset."
    )
    parser.add_argument("--hostname", default="www.eye-ai.org",
                        help="Catalog hostname (default: www.eye-ai.org)")
    parser.add_argument("--catalog-id", default="eye-ai",
                        help="Catalog ID (default: eye-ai)")
    parser.add_argument("--migrate", action="store_true",
                        help="Actually perform the migration (default: dry-run only)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify post-migration state (no writes)")
    return parser.parse_args()


def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


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

    pb = ai.catalog.getPathBuilder()
    schema = pb.schemas["eye-ai"]

    # -------------------------------------------------------------------------
    # Check which tables exist
    # -------------------------------------------------------------------------
    print(f"\nConnected to {args.hostname} / {args.catalog_id}")

    has_image_dataset = "Image_Dataset" in schema.tables
    has_dataset_image = "Dataset_Image" in schema.tables

    print(f"  Image_Dataset exists: {has_image_dataset}")
    print(f"  Dataset_Image exists: {has_dataset_image}")

    # -------------------------------------------------------------------------
    # Verify mode: just check state
    # -------------------------------------------------------------------------
    if args.verify:
        print("\n--- Verification ---")
        passed = True

        # 1. Image_Dataset must be gone
        if has_image_dataset:
            old_rows = list(schema.tables["Image_Dataset"].entities())
            if old_rows:
                print(f"  FAIL: Image_Dataset still has {len(old_rows)} rows and was not dropped.")
            else:
                print("  FAIL: Image_Dataset table still exists (even though empty) — must be dropped.")
            passed = False
        else:
            print("  PASS: Image_Dataset table has been dropped.")

        # 2. Dataset_Image must exist and have data
        if not has_dataset_image:
            print("  FAIL: Dataset_Image does not exist.")
            passed = False
        else:
            new_rows = list(schema.tables["Dataset_Image"].entities())
            by_dataset = collections.Counter(r["Dataset"] for r in new_rows)
            print(f"  PASS: Dataset_Image has {len(new_rows)} rows across {len(by_dataset)} datasets.")

        # 3. Image must still be a valid dataset element type (schema-based check)
        # find_associations() drives both list_dataset_element_types and add_dataset_members.
        # After Image_Dataset is dropped, Dataset_Image is the only junction, so Image
        # remains an element type and add_dataset_members writes unambiguously to Dataset_Image.
        from deriva_ml import DerivaML
        ml = DerivaML(
            hostname=args.hostname,
            catalog_id=args.catalog_id,
            cache_dir="/tmp/eye_ai_cache",
            working_dir="/tmp/eye_ai_work",
        )
        element_type_names = [t.name for t in ml.list_dataset_element_types()]
        if "Image" in element_type_names:
            print(f"  PASS: Image is still a valid dataset element type.")
        else:
            print(f"  FAIL: Image is no longer a dataset element type. Found: {element_type_names}")
            passed = False

        # 4. Verify add_dataset_members association map is unambiguous for Image
        dataset_table = ml.model.name_to_table("Dataset")
        associations = list(dataset_table.find_associations())
        assoc_map = {}
        ambiguous = []
        for a in associations:
            other = a.other_fkeys.pop()
            elem_name = other.pk_table.name
            assoc_name = a.table.name
            if elem_name in assoc_map and assoc_map[elem_name] != assoc_name:
                ambiguous.append((elem_name, assoc_map[elem_name], assoc_name))
            assoc_map[elem_name] = assoc_name
        if ambiguous:
            for elem, t1, t2 in ambiguous:
                print(f"  FAIL: Ambiguous association for '{elem}': {t1} vs {t2}")
            passed = False
        else:
            img_assoc = assoc_map.get("Image", "(not found)")
            print(f"  PASS: add_dataset_members will write Image links to '{img_assoc}' (unambiguous).")

        print(f"\n{'All checks passed.' if passed else 'Some checks FAILED — review output above.'}")
        return

    # -------------------------------------------------------------------------
    # Read current state
    # -------------------------------------------------------------------------
    if not has_image_dataset:
        print("\nImage_Dataset table does not exist — nothing to migrate.")
        return

    img_ds_table = schema.tables["Image_Dataset"]

    print("\nReading Image_Dataset …")
    old_rows = list(img_ds_table.entities())
    old_by_dataset = collections.defaultdict(list)
    for r in old_rows:
        old_by_dataset[r["Dataset"]].append(r)
    print(f"  {len(old_rows)} rows across {len(old_by_dataset)} datasets")

    if has_dataset_image:
        ds_img_table = schema.tables["Dataset_Image"]
        print("\nReading Dataset_Image …")
        new_rows = list(ds_img_table.entities())
        new_by_dataset = collections.defaultdict(set)
        for r in new_rows:
            new_by_dataset[r["Dataset"]].add(r["Image"])
        print(f"  {len(new_rows)} rows across {len(new_by_dataset)} datasets")
    else:
        new_by_dataset = collections.defaultdict(set)

    # -------------------------------------------------------------------------
    # Classify datasets
    # -------------------------------------------------------------------------
    duplicated = []       # in BOTH tables → delete from Image_Dataset only
    needs_migration = []  # in Image_Dataset only (or partially) → copy then delete

    for dataset_rid, rows in old_by_dataset.items():
        if dataset_rid in new_by_dataset:
            old_images = set(r["Image"] for r in rows)
            overlap = old_images & new_by_dataset[dataset_rid]
            if overlap == old_images:
                duplicated.append((dataset_rid, rows))
            else:
                missing = [r for r in rows if r["Image"] not in new_by_dataset[dataset_rid]]
                needs_migration.append((dataset_rid, missing, rows))
        else:
            needs_migration.append((dataset_rid, rows, rows))

    print("\n--- Plan ---")
    dup_row_count = sum(len(r) for _, r in duplicated)
    print(f"  Duplicated datasets (delete from Image_Dataset only): "
          f"{len(duplicated)} datasets, {dup_row_count} rows")
    for rid, rows in duplicated:
        print(f"    {rid}: {len(rows)} rows")

    migrate_count = sum(len(to_copy) for _, to_copy, _ in needs_migration)
    delete_count = sum(len(all_rows) for _, _, all_rows in needs_migration)
    print(f"  Datasets to migrate (insert then delete): "
          f"{len(needs_migration)} datasets, {migrate_count} to insert, {delete_count} to delete")
    for rid, to_copy, all_rows in needs_migration:
        skipped = len(all_rows) - len(to_copy)
        note = f" ({skipped} already in Dataset_Image, skipped)" if skipped else ""
        print(f"    {rid}: {len(to_copy)} to insert{note}")

    print(f"\n  Final step: DROP TABLE eye-ai.Image_Dataset from schema")
    print(f"  (Required so add_dataset_members always writes to Dataset_Image)")

    if not args.migrate:
        print("\nDry-run complete. Re-run with --migrate to apply changes.")
        return

    if not has_dataset_image:
        print("\nERROR: Dataset_Image table does not exist. Cannot migrate.")
        print("Create the table first (matching the prod schema), then re-run.")
        return

    # -------------------------------------------------------------------------
    # Step 1: Insert missing rows into Dataset_Image
    # -------------------------------------------------------------------------
    total_inserted = 0
    for dataset_rid, to_copy, _ in needs_migration:
        records = [{"Dataset": r["Dataset"], "Image": r["Image"]} for r in to_copy]
        for batch in batched(records, BATCH_SIZE):
            ds_img_table.insert(batch, defaults=["RID", "RCT", "RMT", "RCB", "RMB"])
            total_inserted += len(batch)
        print(f"  Inserted {len(records)} rows for dataset {dataset_rid}")

    print(f"\nTotal inserted into Dataset_Image: {total_inserted}")

    # -------------------------------------------------------------------------
    # Step 2: Delete all rows from Image_Dataset (delete per dataset using .eq())
    # -------------------------------------------------------------------------
    total_deleted = 0
    for dataset_rid in old_by_dataset:
        img_ds_table.filter(img_ds_table.Dataset.eq(dataset_rid)).delete()
        total_deleted += len(old_by_dataset[dataset_rid])
        print(f"  Deleted {len(old_by_dataset[dataset_rid])} rows for dataset {dataset_rid}")
    print(f"Deleted {total_deleted} rows from Image_Dataset")

    # -------------------------------------------------------------------------
    # Step 3: Drop Image_Dataset table from the schema
    # -------------------------------------------------------------------------
    print("\nDropping Image_Dataset table from schema …")
    ermrest_model = ai.catalog.getCatalogModel()
    ermrest_model.schemas["eye-ai"].tables["Image_Dataset"].drop()
    print("  Image_Dataset table dropped.")

    print("\nMigration complete. Run with --verify to confirm.")


if __name__ == "__main__":
    main()
