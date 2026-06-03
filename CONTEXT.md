# EYE-AI ML — Context

ML code for the EYE-AI project. `EyeAI` (in [eye_ai/eye_ai.py](eye_ai/eye_ai.py))
is a thin domain subclass of `deriva_ml.DerivaML` that adds eye-ai-specific
routines for working with fundus images, HVF/RNFL reports, and clinical records.

## Glossary

- **EyeAI** — the domain class extending `DerivaML`. Mostly bag-transform helpers
  (`image_tall`, `extract_modality`, `multimodal_wide`, `severity_analysis`) plus a
  few catalog-mutating helpers (`insert_condition_label`). It is a *faithful
  pass-through* constructor over `DerivaML.__init__` — it adds only the eye-ai
  default `hostname` (`www.eye-ai.org`) and `catalog_id` (`eye-ai`).
- **dev catalog** — the `eye-ai` catalog on `dev.eye-ai.org`. It already carries the
  full eye-ai domain schema and data, so read-only integration tests run directly
  against it (no ephemeral-catalog harness needed).
- **subject bag vs image bag** — `multimodal_wide_from_subject_bag` joins a fundus
  *image* dataset bag against a *subject* dataset bag (HVF/RNFL/clinical) as a
  workaround for a deep-FK-traversal download timeout in deriva-ml.
- **data-curation repo** *(planned, not yet created)* — general data-prep
  (image cropping, RETFound directory layout, SVG bounding-box parsing) does **not**
  belong in the `EyeAI` domain class. It is being extracted into a separate
  data-curation repository scaffolded like `eye-ai-model-template`. See
  docs/adr/0002.

## eye-ai-ml is a published, downstream-consumed library

`eye-ai-ml` is **not** a leaf application — it is installed by downstream repos
(e.g. `eye-ai-model-template` pins `eye-ai = { git=..., tag = "v1.5.3" }`, and the
RETFound model repos import `from eye_ai import EyeAI`). Consequences:

- Releases are tagged (`release.sh` → `bump-my-version` → `vX.Y.Z`). Breaking the
  public surface of `EyeAI` breaks tagged downstream consumers.
- `eye-ai` re-exports / pins a compatible `deriva-ml`; the template comment notes
  the `deriva-ml >= 1.42` coupling. Keep the deriva-ml pin coherent across the chain.

## Dependency sourcing (pyproject.toml)

- `deriva-ml` is sourced from GitHub via `[tool.uv.sources]` (the
  `informatics-isi-edu/deriva-ml` repo), **not** PyPI.
- `deriva` (deriva-py) is **intentionally left transitive** — deriva-ml's own lock
  pins it to `deriva-py?branch=deriva-ml`, and we let deriva-ml drive that pin rather
  than declaring a second, possibly-conflicting source in eye-ai. Do not add an
  explicit `[tool.uv.sources]` entry for `deriva`.
- There is no github-pypi index in use; deriva packages come from git and everything
  else from real PyPI.

## EyeAI ↔ deriva-ml coupling

`EyeAI` subclasses `DerivaML` and therefore tracks deriva-ml's API. When deriva-ml
renames/moves/removes a surface EyeAI uses, EyeAI breaks. Known coupling points that
have bitten us (see docs/adr/0001):

- bag denormalization: `DatasetBag.get_denormalized_as_dataframe(...)` (was
  `denormalize_as_dataframe`).
- domain-schema path builder: `self._domain_path()` (was the public `domain_path`
  property; now a private method on `PathBuilderMixin`).
- catalog user lookup: `DerivaML.user_list()` returns `[{"ID", "Full_Name"}]` from
  `public:ERMrest_Client`. **`public` is outside the domain/ML schema search path**,
  so `get_table_as_dict("ERMrest_Client")` will NOT find it — `user_list()` is the
  intended accessor.
