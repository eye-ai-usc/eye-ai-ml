# 0001 — Restore `user_list()` in deriva-ml rather than work around it in EyeAI

- Status: accepted
- Date: 2026-06-03

## Context

`EyeAI.image_tall` maps a grader's `RCB` (created-by user id) to their `Full_Name`
by merging against `self.user_list()`. `DerivaML.user_list()` was **deleted from
deriva-ml** (commit `07fc860`, "dead code — zero callers") as part of an audit sweep.
The audit saw no callers *inside deriva-ml's own tests*, but `EyeAI` is an external
caller it could not see, so the deletion silently broke `image_tall`.

The user records live in `public:ERMrest_Client`. deriva-ml's public table accessors
(`get_table_as_dict`) resolve names through `model.name_to_table()`, whose search
order is `domain_schemas → ml_schema → WWW` — **`public` is deliberately excluded**.
So there is no public deriva-ml API that reaches `ERMrest_Client` today.

We had three options:

1. **Local helper in EyeAI** — `_user_list()` reaching into
   `self.pathBuilder().schemas['public']...fetch()` (raw deriva-py).
2. **Restore `user_list()` in deriva-ml** — re-add the deleted method to
   `PathBuilderMixin`; EyeAI reverts to `self.user_list()`.
3. **Make `name_to_table` search `public`** — change name-resolution semantics
   catalog-wide.

## Decision

Restore `user_list()` in deriva-ml (option 2), placed in `PathBuilderMixin`
(`core/mixins/path_builder.py`) next to the other path-builder accessors. EyeAI calls
`self.user_list()` unchanged.

## Consequences

- Mapping a catalog user id to a name is a **catalog-level concern**, not an eye-ai
  domain concern; every `DerivaML` subclass benefits, and EyeAI does not reach around
  the deriva-ml abstraction into deriva-py.
- **Cost: a cross-repo version dependency.** EyeAI's restored `image_tall` only works
  once eye-ai's lock advances to a deriva-ml commit that contains the restored method.
  The change lands on a deriva-ml feature branch off `main`; eye-ai is re-locked
  (`uv lock --upgrade-package deriva-ml`) to pick it up.
- The restoration is **not** recorded in deriva-ml's release notes — it is a quiet
  re-addition, not a breaking change consumers must migrate around.
- `public` remains outside `name_to_table`'s search path (option 3 rejected as too
  broad); `user_list()` is the one sanctioned door to `ERMrest_Client`.
