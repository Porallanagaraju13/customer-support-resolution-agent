#!/usr/bin/env bash
# Usage: ./run.sh [test|serve|gemini]   (default: serve)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
[ -f .env ] || cp .env.example .env
case "${1:-serve}" in
  test)  exec .venv/bin/python -m pytest -q ;;
  gemini)  # key is read from $GEMINI_API_KEY at runtime; nothing is written to disk
    : "${GEMINI_API_KEY:?export GEMINI_API_KEY first}"
    export LLM_MODE=openai API_KEY_ENV=GEMINI_API_KEY \
      BASE_URL="${BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}" MODEL="${MODEL:-gemini-3.8-flash}"
    exec .venv/bin/uvicorn app.main:create_app --factory --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" ;;
  serve) exec .venv/bin/uvicorn app.main:create_app --factory --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}" ;;
  *) echo "usage: $0 [test|serve|gemini]"; exit 1 ;;
esac
