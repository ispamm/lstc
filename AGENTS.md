# LoSTer 2026 Revision

This repository contains the code and manuscript of a previously reviewed
research paper. Scientific reproducibility is critical.

## General rules

- Do not alter original experimental results unless explicitly requested.
- Do not silently change mathematical definitions or model behavior.
- Before modifying an implementation, identify the corresponding equation
  or description in the manuscript.
- Preserve reproducibility and random seeds.
- Never overwrite original experimental outputs.
- New experiments must be stored separately from legacy results.
- Clearly distinguish:
  1. implementation bugs,
  2. manuscript/code inconsistencies,
  3. methodological improvements,
  4. optional refactoring.
- For every substantive change, explain its scientific consequence.
- Prefer minimal changes until the audit phase is complete.
- Never rewrite manuscript claims merely to match desired conclusions.
- Report negative or unexpected experimental results.
- Do not alter `paper/original_r1/`.
- Treat `paper/original_r1/` as an immutable historical snapshot.
- All manuscript edits must occur in `paper/revision_2026/`.
- Treat `Drafts/` as immutable local historical source material.
- Never add `Drafts/` to Git.
- During the initial audit phase, do not modify model behavior unless explicitly authorized.
- Do not change baseline implementations merely to improve LoSTer comparisons.
- New results must be traceable to configuration, random seed, code commit, and environment.
- Do not make scientific code changes during repository initialization.
