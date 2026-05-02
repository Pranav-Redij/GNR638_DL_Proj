#!/bin/bash
# ============================================================
# setup.bash
# Runs WITH internet on grader's server.
# Steps:
#   1. Clone your GitHub repo
#   2. Create conda env (gnr_project_env, python 3.11)
#   3. Install all dependencies
#   4. Download merged model weights from HuggingFace
# ============================================================

set -e  # exit immediately on any error

# ── CONFIG — update these before submitting ───────────────────
GITHUB_REPO="https://github.com/Pranav-Redij/GNR638_DL_Proj.git"   # your public repo
HF_MODEL="pranavredij/dl-mcq-qwen25-7b"                          # your HF model repo
REPO_DIR="dl-mcq-solver"                                          # folder name after clone
CONDA_ENV="gnr_project_env"
PYTHON_VER="3.11"
# ─────────────────────────────────────────────────────────────

echo "============================================"
echo " DL MCQ Solver - Setup"
echo "============================================"

# ── Step 1: Clone GitHub repo ─────────────────────────────────
echo "[1/4] Cloning repo: $GITHUB_REPO"
if [ -d "$REPO_DIR" ]; then
    echo "  Repo already exists, pulling latest..."
    cd "$REPO_DIR" && git pull && cd ..
else
    git clone "$GITHUB_REPO" "$REPO_DIR"
fi
echo "  Done."

# ── Step 2: Create conda environment ─────────────────────────
echo "[2/4] Creating conda env: $CONDA_ENV (python $PYTHON_VER)"
conda create -y -n "$CONDA_ENV" python="$PYTHON_VER"
echo "  Done."

# ── Step 3: Install dependencies ─────────────────────────────
echo "[3/4] Installing Python packages..."

# Activate env for pip installs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

pip install --upgrade pip -q

pip install -q \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -q \
    transformers>=4.45.0 \
    accelerate>=0.26.0 \
    bitsandbytes>=0.43.0 \
    huggingface_hub \
    peft>=0.10.0 \
    trl>=0.8.0 \
    datasets \
    pandas \
    Pillow \
    opencv-python-headless \
    pytesseract

# Install tesseract OCR system binary
apt-get install -y -q tesseract-ocr

echo "  All packages installed."

# ── Step 4: Download model weights from HuggingFace ──────────
echo "[4/4] Downloading model weights from HF: $HF_MODEL"

MODEL_DIR="./$REPO_DIR/model_weights"
mkdir -p "$MODEL_DIR"

python - <<EOF
from huggingface_hub import snapshot_download
import os

model_dir = "$MODEL_DIR"
hf_model  = "$HF_MODEL"

if os.path.exists(os.path.join(model_dir, "config.json")):
    print(f"  Weights already at {model_dir}, skipping download.")
else:
    print(f"  Downloading {hf_model} → {model_dir} ...")
    snapshot_download(
        repo_id          = hf_model,
        local_dir        = model_dir,
        ignore_patterns  = ["*.msgpack", "*.h5", "flax_model*"],
    )
    print("  Download complete.")
EOF

echo "============================================"
echo " Setup complete!"
echo ""
echo " Next commands to run:"
echo "   conda activate $CONDA_ENV"
echo "   cd $REPO_DIR"
echo "   python inference.py --test_dir <path_to_test_dir>"
echo "============================================"
