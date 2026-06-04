#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

cd "$repo_root"
eval "$(conda shell.bash hook)"
conda activate roboverse

mkdir -p storage_local/sci_out

nohup python "$repo_root/convert_successes.py" storage_local --delete > storage_local/sci_out/convert.log 2>&1 &