#!/bin/bash
# Regenerate the whole paper from the stored first-principles output.
set -e
cd "$(dirname "$0")"
echo "== post-processing DFT =="
(cd dft && python3 postprocess.py > /dev/null)
[ -f dft/kcheck.json ] && (cd dft && python3 postprocess_kcheck.py > /dev/null) || true
echo "== analysis =="
python3 -m rapid.analysis > /dev/null
echo "== figures =="
python3 -m rapid.figures
python3 -m rapid.supplement
python3 -m rapid.numbers
# The manuscript sources are not distributed with the public repository, so
# the typesetting step runs only when a paper/ directory is present.
if [ -f paper/main.tex ]; then
  echo "== latex =="
  cd paper
  for doc in main supplementary; do
    pdflatex -interaction=nonstopmode $doc > /dev/null 2>&1 || true
    bibtex $doc > /dev/null 2>&1 || true
    for _ in 1 2 3; do
      pdflatex -interaction=nonstopmode $doc > /dev/null 2>&1 || true
    done
    echo -n "$doc: "; pdfinfo $doc.pdf | grep -i '^Pages'
    echo "  overfull boxes: $(grep -c 'Overfull' $doc.log || true)"
  done
  cd ..
else
  echo "== latex == skipped, no paper/ sources in this tree"
fi
echo "== archives =="
python3 make_archive.py
