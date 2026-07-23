#!/bin/bash
# Sync Notebooks/ (notebooks, src/, and any local processed_data/) into the
# CodeOcean-shaped bundle at /nrs/ahrens/Ziqiang/Jing_Glia_project/compiled_data_codes,
# rewriting relative data paths for the code/ + data/ split along the way.
#
# Usage: ./sync_to_compiled.sh   (run from anywhere; source path is this script's own folder)
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="/nrs/ahrens/Ziqiang/Jing_Glia_project/compiled_data_codes"

echo "== code: $SRC -> $DEST/code/ =="
rsync -av \
  --exclude='__pycache__' --exclude='.ipynb_checkpoints' --exclude='.DS_Store' \
  --exclude='processed_data' --exclude='sync_to_compiled.sh' \
  "$SRC"/ "$DEST/code/"

if [ -d "$SRC/processed_data" ]; then
  echo "== processed_data: $SRC/processed_data -> $DEST/data/processed_data/ =="
  rsync -av --exclude='.DS_Store' "$SRC/processed_data/" "$DEST/data/processed_data/"
else
  echo "== no local processed_data/ -- leaving $DEST/data/processed_data/ as-is =="
fi

echo "== rewriting data paths in synced code =="
for nb in "$DEST"/code/*.ipynb; do
  [ -e "$nb" ] || continue
  sed -i "s#'processed_data/#'../data/processed_data/#g" "$nb"
done
for py in "$DEST"/code/src/*.py; do
  [ -e "$py" ] || continue
  sed -i "s#'\.\./processed_data/#'../../data/processed_data/#g" "$py"
done

echo "== done. raw_data/ under $DEST/data/ is NRS-only and untouched by this script. =="
echo "== remember to re-run each synced notebook (nbconvert --execute) to verify before considering it done. =="
