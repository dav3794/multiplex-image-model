#!/bin/bash
# Run this once on bury/szary to set up the virtual environment
set -e

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

export PATH="$HOME/.local/bin:$PATH"

cd "$(dirname "$0")"

uv venv ~/venv
source ~/venv/bin/activate
uv pip install -e ".[dev]"

echo "Venv ready at ~/venv"
