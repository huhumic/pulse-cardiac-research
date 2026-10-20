#!/usr/bin/env bash
set -euo pipefail

PULSE_ROOT="${PULSE_ROOT:-/kaggle/working/PULSE}"
PINNED_COMMIT="781e0712ea3b2dd246f86346d73008ecdb73434f"
mkdir -p "$(dirname "$PULSE_ROOT")"

if [[ -f "$PULSE_ROOT/pulse/pulse_seg.py" ]]; then
  echo "[reuse] PULSE already present at $PULSE_ROOT"
else
  echo "[clone] upstream PULSE"
  git clone --quiet https://github.com/BRAIN-Lab-AI/PULSE.git "$PULSE_ROOT"
  git -C "$PULSE_ROOT" checkout --quiet "$PINNED_COMMIT"
fi

# Using >= requirements normally leaves an existing torch installation alone.
python -m pip install -q -r "$PULSE_ROOT/requirements.txt" -r "$(dirname "$0")/../requirements-kaggle.txt"

WEIGHT="$PULSE_ROOT/checkpoints/folds_vdino/fold_0/best_model.pth"
mkdir -p "$(dirname "$WEIGHT")"

if [[ -f "$WEIGHT" && $(wc -c < "$WEIGHT") -gt 300000000 ]]; then
  echo "[reuse] fold_0 checkpoint ($WEIGHT)"
else
  echo "[download] PULSE fold_0 checkpoint"
  curl -fL --retry 3 --connect-timeout 20 -o "$WEIGHT.tmp" \
     'https://github.com/BRAIN-Lab-AI/PULSE/releases/download/v1.0/fold_0.pth'
  if [[ $(wc -c < "$WEIGHT.tmp") -lt 300000000 ]]; then
    echo "Checkpoint download seems incomplete" >&2
    exit 1
  fi
  mv "$WEIGHT.tmp" "$WEIGHT"
fi

python - <<'PY'
import torch
print('torch:', torch.__version__, '| CUDA:', torch.version.cuda)
print('GPU(s):', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY

echo "Setup finished. PULSE_ROOT=$PULSE_ROOT"
