#!/usr/bin/env bash
set -e
export UV_LINK_MODE=copy
echo "Post-create: installing dependencies with uv or requirements.txt"
if command -v uv >/dev/null 2>&1; then
  if [ -f pyproject.toml ]; then
    uv sync || {
      echo "uv sync failed; trying requirements.txt with uv pip"
      [ -f requirements.txt ] && uv pip install -r requirements.txt || true
    }
  elif [ -f requirements.txt ]; then
    uv pip install -r requirements.txt || true
  else
    echo "No pyproject.toml or requirements.txt found; skipping dependency install"
  fi
elif [ -f requirements.txt ]; then
  pip install -r requirements.txt || true
else
  echo "No requirements.txt or uv found; skipping dependency install"
fi
echo "Done."
