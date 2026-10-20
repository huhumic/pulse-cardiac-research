"""Generate the clean .ipynb from reviewed cells, with no saved execution outputs."""
from __future__ import annotations

import json
from pathlib import Path

cells = []

def markdown(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})

markdown("# PULSE ACDC · Kaggle reproduction (fold_0, single checkpoint)\n\n"
         "Do **not** report scores from fold_0 training patients as validation. "
         "Official 4-TTA and fast no-TTA are separate protocols.\n")
code('''%cd /kaggle/working
import torch
print(torch.__version__, torch.version.cuda)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2), "GiB")
''')
markdown("## Clone the independent harness and reuse/download PULSE + fold_0 weight\n"
         "Enable Notebook Internet for the first run; keep ACDC Original Dataset attached under Add Input.\n")
code('''from pathlib import Path
%cd /kaggle/working
import subprocess, os, base64
repo = "https://github.com/huhumic/pulse-cardiac-research.git"
dest = Path("/kaggle/working/pulse-cardiac-research")
env = os.environ.copy()
if not (dest / "README.md").exists():
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_READ_TOKEN")
    except Exception:
        token = None
    if token:
        auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
        env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {auth}"
    subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)], check=True, env=env)
else:
    subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=True, env=env)
%cd /kaggle/working/pulse-cardiac-research
!bash scripts/setup_kaggle.sh
''')
markdown("## Adapt nested NIfTI folders without copying voxel data\n")
code('''!python -m pulse_repro.prepare_acdc --input-root /kaggle/input --output-root /kaggle/working/ACDC_pulse/database
''')
markdown("## First check one official validation patient (2 volumes)\n")
code('''!python -m pulse_repro.evaluate \\
  --acdc-root /kaggle/working/ACDC_pulse/database \\
  --pulse-root /kaggle/working/PULSE --fold 0 \\
  --weights /kaggle/working/PULSE/checkpoints/folds_vdino/fold_0/best_model.pth \\
  --out-dir /kaggle/working/pulse_results/smoke_eval --tta --max-patients 1
''')
markdown("## Full independent fold_0 evaluation (20 patients, ED+ES)\n")
code('''!python -m pulse_repro.evaluate \\
  --acdc-root /kaggle/working/ACDC_pulse/database \\
  --pulse-root /kaggle/working/PULSE --fold 0 \\
  --weights /kaggle/working/PULSE/checkpoints/folds_vdino/fold_0/best_model.pth \\
  --out-dir /kaggle/working/pulse_results/fold0 --tta
''')
markdown("## Benchmark official four-flip TTA and the separate fast no-TTA mode\n")
code('''!python -m pulse_repro.benchmark \\
  --acdc-root /kaggle/working/ACDC_pulse/database \\
  --pulse-root /kaggle/working/PULSE \\
  --weights /kaggle/working/PULSE/checkpoints/folds_vdino/fold_0/best_model.pth \\
  --patient patient014 --phase ED --mode tta --repeats 5 \\
  --out /kaggle/working/pulse_results/speed_tta.json
!python -m pulse_repro.benchmark \\
  --acdc-root /kaggle/working/ACDC_pulse/database \\
  --pulse-root /kaggle/working/PULSE \\
  --weights /kaggle/working/PULSE/checkpoints/folds_vdino/fold_0/best_model.pth \\
  --patient patient014 --phase ED --mode no-tta --repeats 5 \\
  --out /kaggle/working/pulse_results/speed_no_tta.json
''')
markdown("## Optional: 1-epoch training smoke test (this is NOT paper reproduction)\n")
code('''# Uncomment only after reviewing fold0 evaluation and available Kaggle runtime.
# !python -m pulse_repro.train_official \\
#   --acdc-root /kaggle/working/ACDC_pulse/database \\
#   --pulse-root /kaggle/working/PULSE --fold 0 --epochs 1 --batch-size 4 \\
#   --out-dir /kaggle/working/pulse_results/smoke_train_fold0
''')
markdown("## Export reproducibility outputs\n")
code('''from pathlib import Path
import shutil
results = Path("/kaggle/working/pulse_results")
if results.exists():
    archive = shutil.make_archive("/kaggle/working/pulse_reproduction_results", "zip", root_dir=results)
    print("Archive:", archive)
''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
p = Path(__file__).parent / "kaggle_run.ipynb"
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(p)
