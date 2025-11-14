#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo ">>> Using Python: $(command -v python3 || true)"
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3 first."; exit 1
fi

# 1) Create venv if missing
if [ ! -d ".venv" ]; then
  echo ">>> Creating virtual env at .venv"
  python3 -m venv .venv
fi

# 2) Activate venv
# shellcheck disable=SC1091
source .venv/bin/activate
echo ">>> Python in venv: $(python -V)"

# 3) Upgrade base tooling
python -m pip install --upgrade pip setuptools wheel

# 4) Install runtime + dev dependencies with Metal GPU support for macOS M-series
#   PyTorch 2.8.0+ includes native Metal acceleration on macOS
echo ">>> Installing packages..."
pip install \
  pygame \
  numpy \
  torch==2.8.0 \
  torchvision \
  torchaudio \
  matplotlib \
  tqdm \
  tensorboard \
  pydantic \
  black \
  ruff

# 5) Verify Metal GPU support on macOS (M1/M2/M3/M4)
echo ">>> Verifying PyTorch installation..."
python -c "
import torch
import platform
print(f'Python: {platform.python_version()}')
print(f'PyTorch: {torch.__version__}')
print(f'Platform: {platform.machine()}')
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('✓ Metal GPU (MPS) is available and enabled')
else:
    print('⚠ Metal GPU not available; will use CPU')
"

# 6) Freeze exact versions
echo ">>> Writing pinned requirements.txt"
pip freeze | sort > requirements.txt

# 7) Ensure .gitignore exists with sensible defaults
GITIGNORE_CONTENT='
# Virtual env & caches
.venv/
__pycache__/
*.pyc

# Model artifacts & runs
models/
runs/
checkpoints/
logs/

# OS/editor files
.DS_Store
.vscode/
.idea/
'
if [ ! -f ".gitignore" ]; then
  printf "%s" "$GITIGNORE_CONTENT" > .gitignore
  echo ">>> Created .gitignore"
else
  echo ">>> .gitignore already exists (not overwriting)"
fi

echo "data/" >> .gitignore
echo "models/" >> .gitignore

# 7) Quick import check
echo ">>> Verifying pygame + torch imports..."
python - <<'PY'
import pygame, torch
print("ok")
PY

echo ">>> Done. To use the venv now: source .venv/bin/activate"