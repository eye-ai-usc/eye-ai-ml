# 0003 — Rewrite image_tall onto the deriva-ml denormalizer

- Status: accepted
- Date: 2026-06-03

## Context

The interface audit (no-reimplementation pass) found that `EyeAI.image_tall`
reimplemented deriva-ml's denormalization: it hand-rolled a
Subject → Observation → Image → Image_Diagnosis join with three `pd.merge`
calls, duplicating exactly what
`DatasetBag.get_denormalized_as_dataframe(["Subject","Observation","Image","Image_Diagnosis"])`
produces.

The manual join was not laziness — it was load-bearing. `image_tall` labels each
diagnosis with the grader who made it, by merging the diagnosis row's `RCB`
(created-by user id) against `DerivaML.user_list()`. The denormalizer **drops all
system columns** (`RCT`/`RMT`/`RCB`/`RMB`) by default — verified empirically:
`get_denormalized_as_dataframe` returned 0 `RCB` columns — so a naive swap would
silently lose grader identity, the exact feature `user_list()` exists for.

While rewriting we also found the original `image_tall` was **already broken** on
real data, in two ways:
1. `.drop(columns=['RCT','RMT','RMB'])` on the `Image_Diagnosis` frame raises
   `KeyError` when those columns are absent (feature-style tables omit them).
2. `_find_latest_observation` reads `row['date_of_encounter']` (lowercase), but
   the raw `Observation` column is `Date_of_Encounter` (capitalized) — a guaranteed
   `KeyError` on any non-empty frame.

## Decision

Fix the capability gap **at the source in deriva-ml** rather than working around
it in EyeAI: add an opt-in `system_columns: list[str] | None` parameter to the
denormalize chain (deriva-ml PR #283, shipped in v1.45.0) that retains the named
system columns.

Then rewrite `image_tall` to:
- call `get_denormalized_as_dataframe([...], system_columns=["RCB"])` instead of
  the manual merges,
- rename the dotted `Table.column` labels back to the flat names the downstream
  logic and projection expect (mapping `Observation.Date_of_Encounter` →
  `date_of_encounter`, which also fixes bug 2 above),
- keep the grader merge: `Image_Diagnosis.RCB` → `user_list().ID`.

## Consequences

- `image_tall` no longer reimplements deriva-ml denormalization (closes the audit
  finding) and is **more correct** than before — the two latent `KeyError` bugs
  are gone. Verified live: the method runs against the catalog and returns the
  documented 8-column contract.
- **Cross-repo version dependency:** `image_tall` requires deriva-ml ≥ v1.45.0
  (for `system_columns`). eye-ai-ml pins that tag.
- The `system_columns` opt-in keeps the denormalizer's default output clean
  (system columns excluded — see deriva-ml's denormalization rationale) while
  making provenance reachable for the rare caller, like `image_tall`, that needs
  it. The default-off choice is deliberate: system columns multiply per joined
  table and are noise for the wide-table-for-ML use case.
- Behavioral diff-verification against the *old* method is not possible — the old
  method crashes on real data — so correctness is established by the live contract
  test plus the empirical confirmation that `system_columns=["RCB"]` surfaces
  `Image_Diagnosis.RCB`.
