# 0002 — Extract general data-prep out of EyeAI into a data-curation repo

- Status: accepted (executing this session)
- Date: 2026-06-03
- Repo: https://github.com/eye-ai-usc/data-curation (created 2026-06-03, empty)

## Context

`EyeAI` accumulated image data-prep routines that are **not** eye-ai domain logic
and **not** thin wrappers over deriva-ml domain objects:

- `create_cropped_images` — load image assets, optionally crop to a fundus
  bounding box, write into per-diagnosis folders, emit a CSV manifest.
- `create_retfound_image_directory` — call the above 3× to build a RETFound
  train/val/test directory.
- `get_bounding_box` — parse an SVG to extract crop coordinates.
- `get_multimodal_tf_dataset` — incomplete, zero callers.

Three forces converged:

1. **deriva-ml now owns the mechanical core.** `DatasetBag.restructure_assets()`,
   `as_torch_dataset()`, and `as_tf_dataset()` provide asset-to-directory layout,
   label-folder grouping (`target_transform`), and on-placement file conversion
   (`file_transformer`). The hand-rolled loops in `create_cropped_images` largely
   duplicate this.
2. **It's general data-prep, not domain logic.** Cropping / directory layout is a
   data-curation concern reusable across models, not specific to the eye-ai catalog
   schema. It does not belong in the shared `EyeAI` base class.
3. **Consumption is already fragmenting.** `eye-ai-retfound` notebooks call
   `create_retfound_image_directory`; `retfound-severity` rolls its own
   `prepare_data_for_retfound`; `retfound-aireadi` uses its own `DerivaML` shim.

## Decision

Extract the data-prep routines into the **`eye-ai-usc/data-curation` repository**,
scaffolded to look like `eye-ai-model-template` (`src/`, `tests/`, `notebooks/`,
mkdocs `docs/`, hydra-zen + `deriva-ml-run` wiring, tagged releases, pyproject
pinning `eye-ai` + `deriva-ml` from git).

The repo is organized around the **bag-level vs. catalog-level** seam — the natural
boundary for curation work:

- `data_curation/local/` — **bag-based**, offline, no credentials. Holds
  `ImageCurator(ds_bag)`: a thin class over a downloaded `DatasetBag` exposing
  `.to_retfound_directory(...)`, `.cropped_images(...)`. Its methods delegate to
  deriva-ml's `DatasetBag.restructure_assets` / `as_torch_dataset` and supply only
  the eye-ai-specific transforms — SVG bounding-box crop (`file_transformer`),
  Initial-Diagnosis label selection, RID include/exclude filtering. This is where
  `create_cropped_images` / `create_retfound_image_directory` / `get_bounding_box`
  land. (Image cropping for feature creation operates on a local copy anyway.)
- `data_curation/catalog/` — **catalog-level**, takes a `DerivaML` / `EyeAI`
  instance, reads or mutates the live catalog. Empty scaffold now; home for the
  variety of cleaning scripts to come, so they don't bloat `ImageCurator`.

`ImageCurator` takes **only the bag** (cropping needs no live catalog). The design
constraint is that it stays a thin orchestrator over deriva-ml primitives — not a
new junk-drawer base class re-accumulating hand-rolled loops.

## Consequences

- `EyeAI` shrinks to domain logic: bag-transform analytics (`image_tall`,
  `extract_modality`, `multimodal_wide`, `severity_analysis`, the matching helpers)
  and catalog mutation (`insert_condition_label`, `add_multimodal_measurements`).
- **`eye-ai-retfound` notebooks must migrate** to the new repo's entry points —
  a coordinated, cross-repo change. Until then, the cropping routines remain in
  `EyeAI` (API-drift-fixed so they keep working) to avoid breaking tagged consumers.
- `get_multimodal_tf_dataset` (incomplete, zero callers) is removed outright rather
  than relocated.
- The cropping routines are **removed from `eye_ai.py`** (not merely deprecated) as
  part of this work; `eye-ai-retfound`'s two notebooks
  (`RETFound_data.ipynb`, `RETFOUND_DATA_TEMPLATE.ipynb`) are migrated from
  `EA.create_retfound_image_directory(...)` to
  `ImageCurator(bag).to_retfound_directory(...)` in the same effort.
- Sequencing: data-curation depends on `eye-ai` + `deriva-ml`, so the relocation
  lands after eye-ai-ml's API migration is in place. Output is diff-verified against
  the current `create_cropped_images` result on a dev dataset before the EyeAI
  removal is committed.
