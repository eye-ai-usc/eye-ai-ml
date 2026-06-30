# Diagnosis & Severity Vocabularies — Clinical Data Dictionary

- Status: **draft / strawman** — clinical definitions are **provisional**, to be finalized with Dr. Bolo and Dr. Xu
- Date: 2026-06-30
- Catalog: `www.eye-ai.org`, catalog `eye-ai`, schema `eye-ai`
- Scope: clinical reference only — **this document does not modify the catalog** (see §8)

> **Provisional notice.** Every clinical definition, criterion, and threshold in
> this document is a placeholder strawman written to be *argued with*, not a
> settled standard. Nothing here is a clinical authority until reviewed and
> ratified by Dr. Bolo and Dr. Xu.
> Where a definition is unknown, it is marked **TBD — clinical**.

## 1. Purpose & scope

This document defines the **Condition (diagnosis)** and **Severity** vocabularies
used to label glaucoma in the Eye-AI catalog — their **clinical meaning** and
**intended use**, stated **independently of any algorithm, model, or grading
tool**. A term means what a clinician means by it; whether a CNN, an ICD-derived
rule, or a human grader produced a given label does not change the definition of
the label itself.

What this document is:

- The **clinical reference** for the members of `Condition_Label` and
  `Severity_Label` (and the related `Glaucoma_Diagnosis` image-level vocabulary).
- A **strawman conceptual model** for how *condition* and *severity* relate, to be
  refined in the upcoming meeting with Dr. Bolo and Dr. Xu.
- A **record of the problems** in the current vocabularies, grounded in the actual
  catalog data, so the cleanup is driven by evidence rather than opinion.

What this document is **not**:

- It is **not** an algorithm spec, a model card, or a grading protocol.
- It is **not** a numbered ADR — this is an **evolving clinical reference**, not a
  one-time decision record. Decisions that *result* from this work (e.g. renaming
  a term, removing a value) may be captured separately once made.
- It is **not** a mechanism for changing the catalog. Term/schema changes are
  requested through the `data-curation` process (see §8).

### Where this sits in the project (per the `eye-ai-platform` roadmap)

The `eye-ai-platform` repository is the project's front door and repository index.
Per its layering:

- **Schema and vocabulary *changes*** (adding/renaming terms, altering tables) are
  tracked and executed in **`data-curation`** — the catalog-integrity repo. Its
  *"Feature Registration Request"* issue template is the front door for requesting
  a catalog change.
- The **`eye-ai-ml`** library (the `EyeAI` domain class) **reads and writes** these
  vocabularies — e.g. `compute_condition_label()` / `insert_condition_label()` map
  ICD-10 codes into `Condition_Label` values.
- **This document** (in `eye-ai-ml/docs/vocabularies/`) is the **clinical reference
  for what the terms mean.** It informs the `data-curation` requests; it does not
  perform them.

## 2. Current state — every term, exactly as in the catalog

Snapshot taken 2026-06-30 from `www.eye-ai.org` / `eye-ai`. Descriptions and
synonyms are reproduced verbatim; commentary is set off as notes.

### 2.1 `Condition_Label` — 6 terms

> Catalog comment: *"Vocabulary of clinical condition labels used in chart-review
> diagnoses (e.g., glaucoma, diabetic retinopathy)."*

| RID | Name | Description (verbatim) | Synonyms (verbatim) |
|---|---|---|---|
| `2-NKSW` | **GS** | Glaucoma Suspect | `H40.00*`, `H40.01*`, `H40.02*`, `H40.03*`, `H40.04*`, `H40.05*`, `H40.06*` |
| `2-NKSY` | **POAG** | Primary open angle glaucoma | `H40.10*`, `H40.11*`, `H40.12*`, `H40.13*`, `H40.14*`, `H40.15*` |
| `2-NKT0` | **PACG** | Primary angle closure glaucoma | `H40.2*` |
| `2-NKT2` | **Other** | All other conditions (For ICD derived condition label) | — |
| `5-26KY` | **Normal or No dx** | No signs of glaucoma or No diagnosis made | — |
| `6-0A34` | **Unspecified Glaucoma** | Glaucoma present, subtype unspecified (from LAC patient-level chart review). | — |

> **Note — ICD codes in `Synonyms`.** The `Synonyms` field of `GS`, `POAG`, and
> `PACG` is currently being used to store **ICD-10 code patterns** (`H40.*`), not
> human-readable alternate names. This is the lookup key that
> `EyeAI.compute_condition_label()` reverses (its hard-coded `icd_mapping` mirrors
> exactly these patterns: `H40.0x → GS`, `H40.1x → POAG`, `H40.2 → PACG`, else
> `Other`). The codes belong in a defined ICD→condition mapping, not in a
> free-text synonyms list (see §6).

### 2.2 `Severity_Label` — 6 terms

> Catalog comment: *"Vocabulary of severity levels for clinical conditions (e.g.,
> mild, moderate, severe, advanced). Used in chart-review and ICD-derived severity
> classifications."*

| RID | Name | Description (verbatim) | Synonyms |
|---|---|---|---|
| `4-YFWT` | **Mild** | Mild stage | — |
| `4-YFWP` | **Moderate** | Moderate stage | — |
| `4-YFWR` | **Severe** | Severe stage | — |
| `4-YFWW` | **Unspecified/Indeterminate** | Indeterminate stage or stage unspecified | — |
| `4-YFWY` | **GS** | Glaucoma Suspect | — |
| `5-29BJ` | **Normal or No dx** | No signs of glaucoma or No diagnosis made | — |

> **Note.** Three of these six are not severity stages at all: `GS` and
> `Normal or No dx` are *conditions* (they duplicate `Condition_Label` members),
> and `Unspecified/Indeterminate` is a "no stage available" sentinel whose name
> packs a synonym into a slash. Only `Mild`/`Moderate`/`Severe` are genuine
> severity grades. See §3–§4 for the evidence and §6 for proposed fixes.

### 2.3 `Glaucoma_Diagnosis` — 3 terms (consolidated image-level vocabulary)

The image-level diagnostic vocabulary, **created 2026-06-30** as the single
consolidated vocabulary that the three diagnosis levels now share.

> Catalog comment: *"Vocabulary of image-level diagnostic categories for retinal
> images (e.g., referable glaucoma, no glaucoma)."*

| RID | Name | Description (verbatim) | Synonyms (verbatim) |
|---|---|---|---|
| `6-0EQR` | **No Glaucoma** | No Glaucoma | No Referable Glaucoma |
| `6-0EQP` | **Suspected Glaucoma** | Suspected Glaucoma | Referable Glaucoma |
| `6-0EQM` | **Unknown** | Unknown | Ungradable |

**Consumed by** (all foreign-key to `Glaucoma_Diagnosis.Name`, all populated):

| Table | Rows | Level |
|---|---|---|
| `Image_Diagnosis` | 194,204 | image-level (per fundus image) |
| `Observation_Diagnosis` | 7,020 | observation / visit-level |
| `Subject_Diagnosis` | 7,020 | subject-level |

All 194,204 `Image_Diagnosis` rows resolve to exactly these three terms —
**No Glaucoma** 155,279, **Suspected Glaucoma** 38,547, **Unknown** 378 (sum =
194,204) — confirming the 2026-06-30 consolidation is complete with no legacy
values or nulls remaining at the image level.

> **Note — scope of `Glaucoma_Diagnosis` vs `Condition_Label`.** `Glaucoma_Diagnosis`
> is the *image/visit/subject diagnostic category* (a coarse No / Suspected /
> Unknown signal, ~referability), produced by graders and models on images.
> `Condition_Label` is the *clinical chart-review condition* (the specific glaucoma
> subtype: POAG/PACG/GS/...). They are deliberately different axes and should not
> be merged; this document treats `Condition_Label` as the primary "condition"
> vocabulary and notes `Glaucoma_Diagnosis` as the consolidated image-level
> diagnostic signal that feeds it.

### 2.4 Supporting vocabularies (distinct from diagnosis — do not conflate)

These describe *provenance, status, and context* of a diagnosis, not the
diagnosis itself. Listed so the clinical reader knows they exist and why they are
**not** part of the condition/severity cleanup.

- **`Diagnosis_Tag`** (11 terms) — *provenance / study tags*, not diagnoses.
  Examples: `Initial Diagnosis` (from the original dataset), `CNN_Prediction`,
  `Expert_Consensus` (*"Expert consensus diagnosis from 3 expert graders (Benjamin
  Xu, Brandon Wong, Van Nguyen)"*), `Intragrader_Agreement`,
  `GlaucomaSuspect-Training` / `-Validation`, `UI Annotation`. Answers *"who/what
  produced this label, in what study?"*
- **`Diagnosis_Status`** (3 terms) — `Graded`, `Validated`, `Rejected`. Answers
  *"what is the review state of this diagnosis?"*
- **`ICD10_Eye`** — vocabulary of ICD-10 ophthalmic codes; the standardized source
  feeding the ICD-derived condition and severity labels.
- **`Grading_Condition`** — **exists but is currently empty (0 terms)**. Intended
  (per its catalog comment) to *"record the conditions under which a chart label
  was graded, so that the USC and LAC grading contexts remain distinguishable
  within the shared Chart_Label feature table."* Referenced by
  `Execution_Subject_Chart_Label.Grading_Condition`. Flagged here because it is a
  context axis the cleaned-up model may need to populate.

## 3. How the terms are actually used today (the evidence)

The clinical chart-review label lives in the **`Chart_Label` feature on Subject**
— table `Execution_Subject_Chart_Label`, **2,302 rows** — which carries
`Condition_Label` **and** `Severity_Label` as two columns on the same row. The
distinct `(Condition_Label, Severity_Label)` pairs actually present:

| Count | Condition_Label | Severity_Label |
|---:|---|---|
| 698 | GS | **GS** |
| 392 | POAG | Unspecified/Indeterminate |
| 287 | GS | Unspecified/Indeterminate |
| 219 | POAG | Moderate |
| 215 | POAG | Mild |
| 214 | POAG | Severe |
| 107 | Other | Unspecified/Indeterminate |
| 61 | Normal or No dx | Unspecified/Indeterminate |
| 27 | Normal or No dx | **Normal or No dx** |
| 21 | PACG | Unspecified/Indeterminate |
| 20 | PACG | Mild |
| 17 | PACG | Severe |
| 16 | Unspecified Glaucoma | Unspecified/Indeterminate |
| 8 | PACG | Moderate |

Read directly off the data:

- **The only rows carrying a genuine severity grade** (Mild/Moderate/Severe) are
  POAG (648 rows) and PACG (45 rows) — i.e. established glaucoma. Everything else
  is `Unspecified/Indeterminate` or a condition-as-severity sentinel.
- **698 rows are `GS` / `GS`** — the condition *and* the severity are both "GS".
  The severity column is restating the condition.
- **27 rows are `Normal or No dx` / `Normal or No dx`** — same pattern at the
  no-disease end.
- **`Unspecified/Indeterminate` is the catch-all** across every condition (≈900
  rows), used wherever no real stage applies.

## 4. Identified problems (what's broken and why)

1. **`Severity_Label` contains values that are really conditions, not stages.**
   `GS` (`4-YFWY`) and `Normal or No dx` (`5-29BJ`) duplicate `Condition_Label`
   members and make the severity column do double duty. The data proves it: 698
   `GS`/`GS` rows and 27 `Normal`/`Normal` rows where severity simply echoes
   condition.
2. **`Unspecified/Indeterminate` packs a synonym into the name via a slash.** It
   conflates two ideas ("indeterminate stage" vs "stage not recorded") in one
   label and functions as a dumping ground for any condition with no graded stage.
3. **Severity descriptions are circular and thin.** "Mild stage", "Moderate
   stage", "Severe stage" restate the name and carry **no clinical criteria** — no
   VF MD thresholds, no structural criteria, nothing a grader could apply
   reproducibly.
4. **Condition descriptions are better but lean on ICD codes stuffed in
   `Synonyms`.** The `H40.*` patterns in `GS`/`POAG`/`PACG` synonyms are a
   machine lookup key masquerading as alternate names; the clinical definition of
   each condition is mostly carried by the (short) Description, not a proper
   criteria statement.
5. **True severity only meaningfully applies to established glaucoma
   (POAG/PACG).** Grading a "suspect" or a "normal" eye on a Mild/Moderate/Severe
   scale is a category error — which is exactly why those rows fall back to
   condition-as-severity or `Unspecified`.

## 5. Proposed conceptual model (the strawman to discuss)

> **Provisional.** This section is the strawman to be confirmed/revised with
> Dr. Bolo and Dr. Xu.

### 5.1 Two separate axes

- **Condition** = *which glaucoma, or none* — the diagnosis. Candidate members
  (from current `Condition_Label`): `Normal or No dx`, `GS` (glaucoma suspect),
  `POAG`, `PACG`, `Unspecified Glaucoma`, `Other`.
- **Severity** = *stage of established glaucomatous disease* —
  `Mild` / `Moderate` / `Severe` — **applicable only when a glaucomatous condition
  is present**. The non-stage values (`GS`, `Normal or No dx`) move **out** of
  severity; the "no stage available" case is represented explicitly (see §6),
  not by smuggling a condition into the severity column.

### 5.2 Intended relationship: separate but conditional

Severity is a **further attribute of an established glaucoma diagnosis**, not an
independent label. You first have a *condition*; *severity* refines it **only**
when that condition is glaucoma (POAG/PACG, and possibly `Unspecified Glaucoma`).

This directly answers **Professor Carl's question** — *"is there such a thing as 'mild
glaucoma', or is it glaucoma + a separate severity attribute?"* — with a proposed
answer: **separate but conditional.** "Mild glaucoma" = condition `POAG` (or
`PACG`) + severity `Mild`; severity is not baked into the condition term.
**Pending clinical confirmation.**

Corollaries to confirm in the meeting:

- A **suspect (`GS`)** has, by definition, no established disease to stage — so
  does a suspect have a severity at all? (Strawman: **no** — severity is
  undefined/not-applicable for `GS`.)
- A **normal** eye has no severity. (Strawman: not-applicable.)

### 5.3 Data-model placement

**Current join (as built):**

- `Condition_Label` and `Severity_Label` are **two independent foreign-key columns
  side by side** on the `Chart_Label` feature on Subject
  (`Execution_Subject_Chart_Label`: `Condition_Label` → `Condition_Label.Name`,
  `Severity_Label` → `Severity_Label.Name`). Nothing in the schema ties the two —
  any condition can pair with any severity, which is how `GS`/`GS` and
  `Normal`/`Normal` arose.
- **Severity also stands alone**, decoupled from the chart condition, in the
  **ICD-derived feature** `Execution_Clinical_Records_Glaucoma_Severity`
  (**3,674 rows**; column `ICD_Severity_Label` → `Severity_Label.Name`), whose
  target `Clinical_Records` carries the condition separately in
  `Clinical_Records.ICD_Condition_Label` → `Condition_Label.Name`.

**Future enforcement (to discuss).** Once severity is defined as conditional on a
glaucoma condition, the schema *could* enforce "severity requires a glaucoma
condition" — e.g. severity is non-null only for glaucomatous conditions and
null/not-applicable otherwise, or a constraint/validation at write time in the
`Chart_Label` and `Glaucoma_Severity` features. This is a `data-curation` change,
out of scope for this document beyond recording the intent.

### 5.4 The definitions must serve two label sources (stay algorithm-independent)

The term definitions must hold identically for **both** populated paths:

1. **Chart-review labels** — the `Chart_Label` feature
   (`Execution_Subject_Chart_Label`), assigned by human chart review.
2. **ICD-derived labels** — `Clinical_Records.ICD_Condition_Label` and the
   `Glaucoma_Severity` feature (`Execution_Clinical_Records_Glaucoma_Severity`),
   derived from ICD-10 codes.

A definition that only makes sense for one source is wrong. "Moderate POAG" must
mean the same clinical thing whether a grader wrote it or an ICD rule produced it.

### 5.5 Do not conflate the two senses of "severity"

There are **two unrelated meanings of "severity"** in this codebase; the
vocabulary concerns only the first:

- **Severity grade (this document)** — `Severity_Label`: the clinical stage of
  disease (Mild/Moderate/Severe).
- **Laterality severity (different concept)** — `EyeAI.severity_analysis()` in the
  `eye-ai-ml` library computes *which eye is worse* (left vs right) from RNFL
  thickness, HVF MD, and CDR, and flags `Severity_Mismatch`. This is **not** the
  `Severity_Label` vocabulary and must never be confused with it.

The doc states this explicitly so the two are never merged in code or
conversation.

## 6. Naming cleanup proposals

> **Provisional.** Names and descriptions below are placeholders for discussion;
> clinical criteria are **TBD — clinical**, pending Dr. Bolo and Dr. Xu.

### 6.1 `Severity_Label`

| Current term | Proposed action | Proposed name | Proposed description (placeholder) |
|---|---|---|---|
| `Mild` (`4-YFWT`) | **Keep**, add real criteria | `Mild` | *Mild-stage glaucomatous damage.* **TBD — clinical** (VF MD threshold / structural criteria). |
| `Moderate` (`4-YFWP`) | **Keep**, add real criteria | `Moderate` | *Moderate-stage glaucomatous damage.* **TBD — clinical**. |
| `Severe` (`4-YFWR`) | **Keep**, add real criteria | `Severe` | *Severe / advanced-stage glaucomatous damage.* **TBD — clinical**. |
| `Unspecified/Indeterminate` (`4-YFWW`) | **Split & rename** | `Not Staged` (and possibly a separate `Indeterminate`) | Distinguish *"glaucoma present, stage not recorded"* from *"stage genuinely indeterminate."* Remove the slash. |
| `GS` (`4-YFWY`) | **Retire from severity** | — | `GS` is a condition, not a stage; represent via `Condition_Label`. Migrate existing 698+287 rows (see migration note). |
| `Normal or No dx` (`5-29BJ`) | **Retire from severity** | — | A condition / absence of disease, not a stage; represent via `Condition_Label`. Migrate existing rows. |

> **Migration note (for `data-curation`, not this doc):** retiring severity `GS`
> and `Normal or No dx` requires re-mapping the existing `Chart_Label` rows
> (698 `GS`/`GS`, 287 `GS`/`Unspecified`, 61+27 Normal rows) so the condition is
> preserved and severity becomes not-applicable/not-staged. Counts in §3.

### 6.2 `Condition_Label`

| Current term | Proposed action | Note |
|---|---|---|
| `GS`, `POAG`, `PACG` | Keep names; **add proper clinical definitions** | Move the `H40.*` ICD patterns **out of `Synonyms`** into a defined `ICD10_Eye → Condition_Label` mapping (the mapping `compute_condition_label()` already hard-codes). `Synonyms` should hold human alternate names only. |
| `Normal or No dx` | Consider clearer split | "Normal" (no disease) vs "No dx" (not assessed) may warrant separation — **TBD — clinical**. |
| `Unspecified Glaucoma` | Keep | Glaucoma present, subtype unspecified (LAC patient-level). Confirm it is eligible for severity grading. |
| `Other` | Keep | Catch-all for non-glaucoma ICD-derived conditions. |

### 6.3 Severity naming precision

Per Dr. Xu's caution (§7), `Mild/Moderate/Severe` may be ambiguous without a named
basis. Options to decide in the meeting: a neutral name vs. a basis-qualified name
(e.g. `HPA_severity` vs `CMS_severity`). The vocabulary name should make the
**basis of staging unambiguous** so two different staging systems are never
silently mixed in one column.

### 6.4 New term needed for the GAMMA mapping

Dr. Kyle's **GAMMA** mapping will require a severity term **"Moderate-to-Severe"**
(per Dr. Bolo's mapping, GAMMA's **"Progressive"** category maps to a
**Moderate-to-Severe** severity band, which needs to be added to the
vocabulary). Open question for the
meeting: does a `Moderate-to-Severe` band fit the cleaned-up Mild/Moderate/Severe
scheme as an additional member, or should cross-dataset band collapses
(GAMMA/GLEAM) be handled as a **mapping layer** on top of canonical
Mild/Moderate/Severe rather than as new vocabulary terms? Flagged so the cleanup
doesn't lock GAMMA/GLEAM out.

## 7. Open clinical questions for Dr. Bolo & Dr. Xu (for the meeting)

1. **Severity criteria.** What clinical criteria define `Mild` / `Moderate` /
   `Severe`? HPA (Hodapp-Parrish-Anderson) staging? VF MD thresholds? Structural
   (RNFL/CDR) criteria, or a combination? (Dr. Bolo.)
2. **Separate but conditional?** Confirm severity should be a separate attribute
   that applies **only** when an established glaucoma condition is present
   (Professor Carl's question; strawman in §5.2).
3. **Naming basis.** How to name severity so its meaning is unambiguous — neutral
   name vs. basis-qualified (`HPA_severity` vs `CMS_severity` vs a neutral label)
   — per Dr. Xu's caution about silently mixing staging systems.
4. **Remove non-stage values?** Confirm retiring `GS` and `Normal or No dx` from
   `Severity_Label` (they are conditions, not stages).
5. **Suspects and severity.** Does a glaucoma suspect (`GS`) have a severity at
   all? (Strawman: no — not applicable.)
6. **GLEAM / GAMMA.** What, if anything, must the cleaned-up vocabulary provide so
   the GLEAM and GAMMA mappings land cleanly — notably GAMMA's `Moderate-to-Severe`
   band (§6.4) — without distorting the canonical clinical definitions?

## 8. References / provenance

- This document consolidates the read-only investigation captured in
  `tacit-knowledge.md` (the diagnosis/condition/severity landscape review,
  2026-06-30).
- **Catalog objects referenced** (schema `eye-ai`): vocabularies `Condition_Label`,
  `Severity_Label`, `Glaucoma_Diagnosis`, `Diagnosis_Tag`, `Diagnosis_Status`,
  `ICD10_Eye`, `Grading_Condition`; feature/association tables
  `Execution_Subject_Chart_Label` (Chart_Label feature on Subject),
  `Execution_Clinical_Records_Glaucoma_Severity` (Glaucoma_Severity feature on
  Clinical_Records), `Clinical_Records`, `Image_Diagnosis`, `Observation_Diagnosis`,
  `Subject_Diagnosis`.
- **Library code referenced** (`eye-ai-ml/eye_ai/eye_ai.py`):
  `compute_condition_label()`, `insert_condition_label()`, `severity_analysis()`
  (laterality — see §5.5).
- **Roadmap context:** repository layering per `eye-ai-usc/eye-ai-platform`
  (front door / repository index).
- **Process note — no catalog edits from this document.** Any term rename,
  removal, description change, or new term (§6) is a catalog change and must be
  requested through **`data-curation`** via its *"Feature Registration Request"*
  issue template. This document is the clinical rationale that such requests cite;
  it does not itself mutate the catalog.

---

*Draft for review. All clinical definitions provisional pending Dr. Bolo and
Dr. Xu. Counts and term contents reflect the catalog state on 2026-06-30.*
