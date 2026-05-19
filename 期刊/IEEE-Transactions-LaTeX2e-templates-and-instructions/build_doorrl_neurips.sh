#!/usr/bin/env bash
set -euo pipefail

MAIN="doorrl_neurips_main"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
  bibtex "${MAIN}"
  pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
  pdflatex -interaction=nonstopmode -halt-on-error "${MAIN}.tex"
else
  echo "No LaTeX compiler found."
  echo "Install TeXLive/latexmk or provide pdflatex+bibtex, then run:"
  echo "  bash build_doorrl_neurips.sh"
  exit 1
fi

echo "Built ${MAIN}.pdf"
