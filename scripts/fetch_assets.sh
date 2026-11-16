#!/usr/bin/env bash
# Downloads the Natural Earth map data (public domain) and the male Piper voice. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
NE=data/naturalearth; mkdir -p "$NE" models
BASE=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson
for f in ne_50m_admin_0_countries ne_50m_lakes ne_50m_rivers_lake_centerlines ne_50m_populated_places_simple; do
  [ -s "$NE/$f.geojson" ] || curl -fsSL --retry 3 -o "$NE/$f.geojson" "$BASE/$f.geojson"
done
VOICE="${PIPER_VOICE:-en_US-ryan-high}"
# en_US-ryan-high -> en/en_US/ryan/high/
LANG_CODE="${VOICE%%-*}"; REST="${VOICE#*-}"; SPEAKER="${REST%-*}"; QUALITY="${REST##*-}"
HF="https://huggingface.co/rhasspy/piper-voices/resolve/main/${LANG_CODE%%_*}/${LANG_CODE}/${SPEAKER}/${QUALITY}"
for ext in onnx onnx.json; do
  [ -s "models/$VOICE.$ext" ] || curl -fsSL --retry 3 -o "models/$VOICE.$ext" "$HF/$VOICE.$ext"
done
ls -la "$NE" models
