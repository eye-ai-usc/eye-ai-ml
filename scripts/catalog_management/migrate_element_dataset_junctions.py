"""
Migrate element-dataset membership from old {Element}_Dataset junction tables
(Element has FK → Dataset) to new Dataset_{Element} junction tables
(Dataset has FK → Element), then drop the old tables.

This is the generalised version of migrate_image_dataset_to_dataset_image.py.
Do NOT use this script for Image — that migration has already been completed.

Background
----------
DerivaML uses Dataset_{Element} as the standard naming convention.
Old catalogs used {Element}_Dataset instead.  When both exist,
add_dataset_members uses find_associations() which produces a non-deterministic
mapping (last one wins in the dict comprehension), so new members can end up
in either table.  Dropping the old table makes behaviour consistent.

Affected tables (as of 2026-04-06)
------------------------------------
  dev.eye-ai.org:
    Subject_Dataset    : 233,399 rows  (needs migration → Dataset_Subject)
    Dataset_Subject    :     145 rows  (already has some data)
    Dataset_Observation:     355 rows  (only new-style exists — nothing to migrate)

  www.eye-ai.org:
    Subject_Dataset    : 240,762 rows  (needs migration → Dataset_Subject)
    Observation_Dataset:   7,850 rows  (needs migration → Dataset_Observation)

Usage
-----
# Dry-run (default) — safe, read-only:
    uv run python scripts/catalog_management/migrate_element_dataset_junctions.py \\
        --hostname dev.eye-ai.org

# Migrate a specific element:
    uv run python scripts/catalog_management/migrate_element_dataset_junctions.py \\
        --hostname dev.eye-ai.org --elements Subject --migrate

# Migrate all default elements:
    uv run python scripts/catalog_management/migrate_element_dataset_junctions.py \\
        --hostname www.eye-ai.org --migrate

# Verify after migration:
    uv run python scripts/catalog_management/migrate_element_dataset_junctions.py \\
        --hostname dev.eye-ai.org --verify
"""

import argparse
import collections
import logging

from eye_ai.eye_ai import EyeAI

# Default elements to handle (Image intentionally excluded — already migrated)
DEFAULT_ELEMENTS = ["Subject", "Observation"]
BATCH_SIZE = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate {Element}_Dataset → Dataset_{Element} and drop old tables."
    )
    parser.add_argument("--hostname", default="www.eye-ai.org",
                        help="Catalog hostname (default: www.eye-ai.org)")
    parser.add_argument("--catalog-id", default="eye-ai",
                        help="Catalog ID (default: eye-ai)")
    parser.add_argument("--elements", nargs="+", default=DEFAULT_ELEMENTS,
                        help=f"Element names to migrate (default: {DEFAULT_ELEMENTS}). "
                             "Image is excluded — use migrate_image_dataset_to_dataset_image.py instead.")
    parser.add_argument("--migrate", action="store_true",
                        help="Actually perform the migration (default: dry-run only)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify post-migration state (no writes)")
    return parser.parse_args()


def batched(iterable, n):
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


def migrate_element(ai, element: str, do_migrate: bool, pb) -> dict:
    """Migrate one element type. Returns a summary dict."""
    old_name = f"{element}_Dataset"
    new_name = f"Dataset_{element}"
    schema = pb.schemas["eye-ai"]

    has_old = old_name in schema.tables
    has_new = new_name in schema.tables

    print(f"\n--- {element} ---")
    print(f"  {old_name}: {'EXISTS' if has_old else 'does not exist'}", end="")
    if has_old:
        old_count = len(list(schema.tables[old_name].entities()))
        print(f", {old_count} rows", end="")
    print()
    print(f"  {new_name}: {'EXISTS' if has_new else 'does not exist'}", end="")
    if has_new:
        new_count = len(list(schema.tables[new_name].entities()))
        print(f", {new_count} rows", end="")
    print()

    if not has_old:
        print(f"  Nothing to migrate — {old_name} does not exist.")
        return {"element": element, "status": "skipped"}

    old_table = schema.tables[old_name]
    old_rows = list(old_table.entities())
    old_by_dataset = collections.defaultdict(list)
    for r in old_rows:
        old_by_dataset[r["Dataset"]].append(r)

    if has_new:
        new_rows = list(schema.tables[new_name].entities())
        new_by_dataset = collections.defaultdict(set)
        for r in new_rows:
            new_by_dataset[r["Dataset"]].add(r[element])
    else:
        new_by_dataset = collections.defaultdict(set)

    # Classify
    duplicated = []
    needs_migration = []
    for dataset_rid, rows in old_by_dataset.items():
        if dataset_rid in new_by_dataset:
            old_elems = set(r[element] for r in rows)
            overlap = old_elems & new_by_dataset[dataset_rid]
            if overlap == old_elems:
                duplicated.append((dataset_rid, rows))
            else:
                missing = [r for r in rows if r[element] not in new_by_dataset[dataset_rid]]
                needs_migration.append((dataset_rid, missing, rows))
        else:
            needs_migration.append((dataset_rid, rows, rows))

    dup_count = sum(len(r) for _, r in duplicated)
    migrate_count = sum(len(to_copy) for _, to_copy, _ in needs_migration)
    delete_count = sum(len(all_rows) for _, _, all_rows in needs_migration)

    print(f"  Plan:")
    print(f"    Duplicated (delete from {old_name} only): "
          f"{len(duplicated)} datasets, {dup_count} rows")
    print(f"    To migrate (insert into {new_name}, delete from {old_name}): "
          f"{len(needs_migration)} datasets, {migrate_count} to insert, {delete_count} to delete")
    print(f"    Final step: DROP TABLE eye-ai.{old_name}")

    if not do_migrate:
        return {"element": element, "status": "dry-run",
                "to_insert": migrate_count, "to_delete": len(old_rows)}

    # Create new table if needed
    if not has_new:
        print(f"  Creating {new_name} via add_dataset_element_type('{element}')...")
        table = ai.add_dataset_element_type(element)
        print(f"  Created: {table.schema.name}.{table.name}")
        # Refresh schema reference
        schema = pb.schemas["eye-ai"]

    new_table = schema.tables[new_name]

    # Step 1: Insert missing rows
    total_inserted = 0
    for dataset_rid, to_copy, _ in needs_migration:
        records = [{"Dataset": r["Dataset"], element: r[element]} for r in to_copy]
        for batch in batched(records, BATCH_SIZE):
            new_table.insert(batch, defaults=["RID", "RCT", "RMT", "RCB", "RMB"])
            total_inserted += len(batch)
        print(f"  Inserted {len(records)} rows for dataset {dataset_rid}")
    print(f"  Total inserted into {new_name}: {total_inserted}")

    # Step 2: Delete all rows from old table (per dataset)
    total_deleted = 0
    for dataset_rid in old_by_dataset:
        old_table.filter(old_table.Dataset.eq(dataset_rid)).delete()
        total_deleted += len(old_by_dataset[dataset_rid])
    print(f"  Deleted {total_deleted} rows from {old_name}")

    # Step 3: Drop old table
    print(f"  Dropping {old_name} from schema...")
    ermrest_model = ai.catalog.getCatalogModel()
    ermrest_model.schemas["eye-ai"].tables[old_name].drop()
    print(f"  {old_name} dropped.")

    return {"element": element, "status": "migrated",
            "inserted": total_inserted, "deleted": total_deleted}


def verify_element(ai, element: str, pb) -> bool:
    """Verify one element type post-migration. Returns True if all checks pass."""
    old_name = f"{element}_Dataset"
    new_name = f"Dataset_{element}"
    schema = pb.schemas["eye-ai"]
    passed = True

    print(f"\n--- {element} ---")

    # 1. Old table must be gone
    if old_name in schema.tables:
        rows = list(schema.tables[old_name].entities())
        if rows:
            print(f"  FAIL: {old_name} still has {len(rows)} rows and was not dropped.")
        else:
            print(f"  FAIL: {old_name} still exists (empty) — should be dropped.")
        passed = False
    else:
        print(f"  PASS: {old_name} has been dropped.")

    # 2. New table must exist with data
    if new_name not in schema.tables:
        print(f"  FAIL: {new_name} does not exist.")
        passed = False
    else:
        new_rows = list(schema.tables[new_name].entities())
        by_dataset = collections.Counter(r["Dataset"] for r in new_rows)
        print(f"  PASS: {new_name} has {len(new_rows)} rows across {len(by_dataset)} datasets.")

    # 3. Element must still be a valid dataset element type
    element_type_names = [t.name for t in ai.list_dataset_element_types()]
    if element in element_type_names:
        print(f"  PASS: {element} is still a valid dataset element type.")
    else:
        print(f"  FAIL: {element} is no longer a dataset element type.")
        passed = False

    # 4. Association map must be unambiguous
    dataset_table = ai.model.name_to_table("Dataset")
    assoc_map = {}
    ambiguous = []
    for a in dataset_table.find_associations():
        other = a.other_fkeys.pop()
        elem_name = other.pk_table.name
        assoc_name = a.table.name
        if elem_name in assoc_map and assoc_map[elem_name] != assoc_name:
            ambiguous.append((elem_name, assoc_map[elem_name], assoc_name))
        assoc_map[elem_name] = assoc_name
    elem_ambiguous = [(e, t1, t2) for e, t1, t2 in ambiguous if e == element]
    if elem_ambiguous:
        for e, t1, t2 in elem_ambiguous:
            print(f"  FAIL: Ambiguous association for '{e}': {t1} vs {t2}")
        passed = False
    else:
        assoc = assoc_map.get(element, "(not found)")
        print(f"  PASS: add_dataset_members will write {element} links to '{assoc}' (unambiguous).")

    return passed


def main() -> None:
    args = parse_args()

    if "Image" in args.elements:
        print("WARNING: Image is excluded from this script — it was already migrated.")
        print("         Use migrate_image_dataset_to_dataset_image.py for Image.")
        args.elements = [e for e in args.elements if e != "Image"]
        if not args.elements:
            return

    ai = EyeAI(
        hostname=args.hostname,
        catalog_id=args.catalog_id,
        cache_dir="/tmp/eye_ai_cache",
        working_dir="/tmp/eye_ai_work",
        logging_level=logging.INFO,
        deriva_logging_level=logging.ERROR,
    )

    pb = ai.catalog.getPathBuilder()
    print(f"\nConnected to {args.hostname} / {args.catalog_id}")
    print(f"Elements to process: {args.elements}")

    if args.verify:
        print("\n=== Verification ===")
        all_passed = True
        for element in args.elements:
            passed = verify_element(ai, element, pb)
            all_passed = all_passed and passed
        print(f"\n{'All checks passed.' if all_passed else 'Some checks FAILED — review output above.'}")
        return

    results = []
    for element in args.elements:
        result = migrate_element(ai, element, args.migrate, pb)
        results.append(result)

    print("\n=== Summary ===")
    for r in results:
        if r["status"] == "skipped":
            print(f"  {r['element']}: skipped (old table not found)")
        elif r["status"] == "dry-run":
            print(f"  {r['element']}: dry-run — {r['to_insert']} to insert, {r['to_delete']} to delete")
        else:
            print(f"  {r['element']}: migrated — {r['inserted']} inserted, {r['deleted']} deleted")

    if not args.migrate:
        print("\nDry-run complete. Re-run with --migrate to apply changes.")


if __name__ == "__main__":
    main()
