#!/bin/bash

echo "Running..."
echo "Safe to close the terminal/VSCode; job will continue."

# Ignore hangup so the script survives when the terminal closes.
trap 'echo "Terminal closed — continuing in background..."' HUP

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found; skipping GPU summary."
fi

# === Activate Conda environment ===
# Load conda into the shell (required for non-interactive scripts)
eval "$(conda shell.bash hook)"
conda activate ray

# Create the output logs directory if it doesn't exist
mkdir -p storage_local/sci_out

# Get the current date and time in the desired format
current_datetime=$(date +"%Y_%m_%d__%H_%M_%S")

# Set the log file name
log_file="storage_local/sci_out/log_${current_datetime}.out"

# List of Python scripts to run
scripts=(
    "random_agent.py"
)

# Run each Python script in the background and append output to the single log file
pids=()
for script in "${scripts[@]}"; do
    echo "===== Launching $script =====" | tee -a "$log_file"
    nohup python "$script" >> "$log_file" 2>&1 &
    pid=$!
    pids+=("$pid")
    echo "Started $script (PID $pid). Output streaming to $log_file" | tee -a "$log_file"
done

# Print the location of the log file and background job info
echo "Background job PIDs: ${pids[*]}"
echo "Tail logs with: tail -f $log_file"
echo "All output is being saved to: $log_file"
