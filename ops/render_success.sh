#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

cd "$repo_root"
eval "$(conda shell.bash hook)"
conda activate roboverse

mkdir -p storage_local/sci_out

nohup python "$repo_root/render_success.py" \
	storage_local/ant__20260604_0105__13499968__train_ppo__single/successes/success_t000809232_ep1094_len346.json \
	> storage_local/sci_out/render.log 2>&1 &