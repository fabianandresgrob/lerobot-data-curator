#!/usr/bin/env bash
# Setup script for lerobot-data-curator.
# Creates a single virtual environment in score_lerobot_episodes/.venv
# and installs both score_lerobot_episodes and I-FailSense into it.
set -e

# Check uv is available
if ! command -v uv &>/dev/null; then
    echo "Error: uv not found. Install it with: pip install uv"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/score_lerobot_episodes/.venv"
PYTHON="$VENV/bin/python"

echo "==> Creating virtual environment at $VENV"
cd "$REPO_ROOT/score_lerobot_episodes"
uv venv .venv --python python3.10

echo "==> Installing score_lerobot_episodes"
uv pip install -e . --python "$PYTHON"

echo "==> Installing I-FailSense (--no-deps to preserve torch version)"
uv pip install -e "$REPO_ROOT/I-FailSense" --no-deps --python "$PYTHON"

echo "==> Installing I-FailSense dependencies (excluding torch to avoid version conflict)"
uv pip install peft scikit-learn seaborn matplotlib --python "$PYTHON"

echo "==> Installing huggingface_hub for weight auto-download"
uv pip install huggingface_hub --python "$PYTHON"

echo "==> Installing UI dependencies"
uv pip install streamlit plotly --python "$PYTHON"

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source score_lerobot_episodes/.venv/bin/activate"
echo ""

# Check for system ffmpeg
if ! command -v ffprobe &>/dev/null; then
    echo "WARNING: ffprobe not found. The technical scorer requires system ffmpeg."
    echo "Install it with:"
    echo "  sudo apt-get install ffmpeg      # Ubuntu/Debian"
    echo "  conda install -c conda-forge ffmpeg  # Conda"
    echo ""
fi

echo "Then filter a dataset with:"
echo "  python curate_dataset.py --repo_id your/dataset --task_description '...' --output filtered/"
