# DOOR-RL NeurIPS LaTeX Draft

This folder follows the existing IEEE template directory style, but adds a NeurIPS-oriented manuscript:

- Main TeX: `doorrl_neurips_main.tex`
- Bibliography: `doorrl_refs.bib`
- Build helper: `build_doorrl_neurips.sh`

## Build

From this directory:

```bash
bash build_doorrl_neurips.sh
```

The script uses `latexmk` if available, otherwise falls back to `pdflatex + bibtex`.

Current machine note: the `doorrl` Python environment has paper-helper Python packages installed (`pylatexenc`, `bibtexparser`, `latexcodec`), but this machine does not currently expose a LaTeX compiler (`pdflatex`, `bibtex`, `latexmk`, or `tectonic`). Install TeXLive/latexmk or run on a machine that already has a LaTeX toolchain.

## Figure Inputs

The main TeX references paper-ready closure figures from:

```text
../paper_assets/neurips_closure_2026-04-28/figures/
```

Referenced figures:

- `fig_selection_diagnostics_bars`
- `fig_relation_semantics_ablation`
- `fig_budget_sensitivity_12_4_vs_10_6`

Extensions are omitted in `\includegraphics` so LaTeX can pick PDF or PNG depending on what exists.

## Current Manuscript State

The draft currently includes:

- Type competition problem framing
- Learned ego-conditioned top-K selector definition
- Typed-budget dynamic/relation abstraction
- Selection diagnostics: CDR, MissRate, WastedRel
- Stage0 nuScenes table
- nuPlan 50k Stage1 table
- `wm_naive` 50k Stage1 closure paragraph
- `wm_naive` 50k selection diagnostic table
- Relation feature ablation
- Budget sensitivity
- Reproducibility/artifact map

Pending updates:

- Replace the generic `article` class with the official NeurIPS style file when the target year/template is fixed.
- Add conceptual Figure 1 for type competition vs typed budget.
