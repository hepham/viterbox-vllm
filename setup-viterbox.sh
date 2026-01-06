#!/bin/bash
# Setup script for chatterbox-vllm with Viterbox support on WSL2/Linux

set -e

echo "=== Setting up chatterbox-vllm with Viterbox support ==="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Create venv and install
echo "Creating virtual environment..."
uv venv
source .venv/bin/activate

echo "Installing dependencies..."
uv sync

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run Viterbox example:"
echo "  source .venv/bin/activate"
echo "  python example-tts-viterbox.py"
echo ""
