#!/usr/bin/env bash
set -euo pipefail
DEST="${1:-external/GMTKN55}"
if [ -e "$DEST" ]; then
  printf 'Refusing to overwrite existing destination: %s\n' "$DEST" >&2
  exit 1
fi
git clone --depth 1 --branch v2 https://github.com/grimme-lab/GMTKN55.git "$DEST"
printf '\nReview this exact upstream commit and the source license before study freeze:\n'
git -C "$DEST" rev-parse HEAD
printf 'No .res shell script was executed. No chemistry calculation was launched.\n'
