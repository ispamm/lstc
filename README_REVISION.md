# LoSTer 2026 revision workspace

This repository is the ISPAMM working fork for the 2026 revision of
"Concrete Dense Network for Long-Sequence Time Series Clustering" (LoSTer).

ispamm/lstc is a GitHub fork of jrtaloma/lstc and preserves attribution
to the original repository.

No scientific changes were made during repository initialization.

## Repository provenance

| Reference | Purpose |
| --- | --- |
| `origin` | ISPAMM working fork: https://github.com/ispamm/lstc |
| `upstream` | Original repository: https://github.com/jrtaloma/lstc |
| `original-code` | Annotated tag identifying the untouched starting code, commit `b594b650f68ebed80436b8de7313fa1536d3d95c` |
| `revision-2026` | Branch for all future scientific revision work |

The tag message is "Original LoSTer code state before 2026 revision".
Push new work to `origin` only. Never push to `upstream`.
Existing repository directories and relative paths remain intact.

## Workspace

| Path | Purpose |
| --- | --- |
| `Drafts/` | Immutable, local, untracked historical source archive |
| `paper/original_r1/` | Immutable source snapshot extracted from the rejected Pattern Recognition R1 Overleaf project |
| `paper/revision_2026/` | Working manuscript for the new revision; the only manuscript tree that may later be edited |
| `reviews/` | Public, high-level editorial history; no private correspondence |
| `analysis/audits/` | Scientific, mathematical, implementation, and reproducibility audits |
| `analysis/literature/` | Literature analysis for the revision |
| `analysis/statistics/` | Statistical analysis for the revision |
| `experiments_new/configs/` | Configurations for new 2026 experiments |
| `experiments_new/logs/` | Generated experiment logs, ignored by Git |
| `experiments_new/results/` | Generated results and caches, ignored by Git |
| `experiments_new/scripts/` | Reproducibility scripts for new experiments |
| `docs/` | Revision documentation |

The Overleaf archive was extracted completely, preserving filenames,
directories, and file bytes. Its 12 source files were copied to
`paper/revision_2026/`. Recursive SHA-256 comparison and direct byte comparison
confirmed that both trees and the corresponding ZIP entries were identical
at initialization. PDF figures belonging to the manuscript source are tracked;
historical manuscript and editorial PDFs remain local.

## Local historical source material

The following files are intentionally untracked and remain under `Drafts/`:

- `Pattern_Recognition___Concrete_Dense_Network_for_Long_Sequence_Time_Series_Clustering.zip`:
  latest R1 source supplied as an Overleaf project; the latest R1 source is
  reconstructed from this ZIP in `paper/original_r1/`.
- `PR-D-25-00861.pdf`: original submitted manuscript PDF, not the R1 manuscript.
- `Your Submission PR-D-25-02649R1.pdf`: final editorial/review decision material,
  local-only.

No separate R1 manuscript PDF was found in `Drafts/`.
Do not rename, move, modify, stage, or commit anything in `Drafts/`.
Do not reproduce reviewer correspondence in the public repository.

## Initialization scope

Initialization adds source snapshots, directory placeholders, ignore rules,
and revision documentation only. Original scientific source files and legacy
experimental results are unchanged. No scientific audit, experiments, model
refactoring, or manuscript revisions were performed.

MiKTeX executables `pdflatex` and `latexmk` were found locally. No LaTeX
compilation was attempted and no packages or dependencies were installed.
Follow `AGENTS.md` for future work; never overwrite original outputs.
