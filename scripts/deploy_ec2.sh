#!/usr/bin/env bash

set -euo pipefail

BRANCH="${DEPLOY_BRANCH:-main}"
SERVICE_NAME="${EC2_SYSTEMD_SERVICE:-}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required on the EC2 instance"
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required on the EC2 instance"
  exit 1
fi

echo "Deploying branch: $BRANCH"
git fetch --prune origin
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/process_data.py

if [ -n "$SERVICE_NAME" ]; then
  sudo systemctl restart "$SERVICE_NAME"
  sudo systemctl status "$SERVICE_NAME" --no-pager
else
  echo "EC2_SYSTEMD_SERVICE not set. Skipping service restart."
fi
