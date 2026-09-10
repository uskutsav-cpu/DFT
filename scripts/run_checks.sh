#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m ruff format --check src tests scripts workflows
python -m ruff check src tests scripts workflows
python -m pytest -q -m 'not integration' --cov=refblind --cov-report=term-missing
python -m refblind demo --groups 40 --output "results/check-$(date -u +%Y%m%dT%H%M%SZ)"
