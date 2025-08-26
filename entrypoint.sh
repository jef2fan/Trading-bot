#!/usr/bin/env bash
set -euo pipefail

ROLE="${APP_ROLE:-web}"
if [ "$ROLE" = "worker" ]; then
  exec python -u worker.py
else
  exec python -u app.py
fi
