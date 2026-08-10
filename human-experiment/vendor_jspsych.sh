#!/usr/bin/env bash
# Vendor the jsPsych browser bundles into public/vendor/.
#
# index.html loads jsPsych as plain <script> tags, which needs the browser IIFE
# builds. The node_modules copy committed to this repo was stripped of every
# package's dist/ directory, so those globals never resolved and the only path
# that worked was the local preview server redirecting to a CDN. Pulling the
# published tarballs into public/ makes the experiment self-contained and lets
# any static host (Vercel included) serve it with no run-time CDN dependency.
#
# Usage: bash human-experiment/vendor_jspsych.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/public/vendor"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Keep these pinned to the versions in package.json.
PACKAGES=(
  "jspsych@8.2.3|jspsych.js"
  "@jspsych/plugin-preload@2.1.0|plugin-preload.js"
  "@jspsych/plugin-instructions@2.1.0|plugin-instructions.js"
  "@jspsych/plugin-html-button-response@2.1.0|plugin-html-button-response.js"
)

mkdir -p "$DEST"
cd "$WORK"

for entry in "${PACKAGES[@]}"; do
  spec="${entry%%|*}"
  outname="${entry##*|}"
  echo "fetching $spec"
  tarball="$(npm pack "$spec" --silent)"
  rm -rf package
  tar xzf "$tarball"
  cp package/dist/index.browser.min.js "$DEST/$outname"
  # jsPsych core also ships the stylesheet the page needs.
  if [[ -f package/css/jspsych.css ]]; then
    cp package/css/jspsych.css "$DEST/jspsych.css"
  fi
done

echo ""
echo "vendored into $DEST:"
ls -la "$DEST"
