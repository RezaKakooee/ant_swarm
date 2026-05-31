#!/bin/bash
# Watch the most recent log from py_runner.sh and show live process status.
#
# Usage:
#   bash ops/watch_py.sh          # tail latest log + show process status
#   bash ops/watch_py.sh <pid>    # watch a specific PID

LOG_DIR="storage_local/sci_out"
latest_log=$(ls -t "$LOG_DIR"/log_*.out 2>/dev/null | head -1)

if [ -z "$latest_log" ]; then
    echo "No log files found in $LOG_DIR"
    exit 1
fi

echo "========================================"
echo "Latest log : $latest_log"
echo "========================================"

# Extract PID from the log if not provided
if [ -n "$1" ]; then
    pid="$1"
else
    pid=$(grep -oP '(?<=PID )\d+' "$latest_log" | tail -1)
fi

if [ -n "$pid" ]; then
    if ps -p "$pid" > /dev/null 2>&1; then
        elapsed=$(ps -p "$pid" -o etime= 2>/dev/null | tr -d ' ')
        cmd=$(ps -p "$pid" -o cmd= 2>/dev/null)
        echo "Status     : RUNNING  (PID $pid, elapsed $elapsed)"
        echo "Command    : $cmd"
    else
        echo "Status     : FINISHED or CRASHED  (PID $pid no longer running)"
    fi
else
    echo "Status     : PID not found in log"
fi

echo "========================================"
echo "Tailing log (Ctrl+C to stop)..."
echo ""
tail -f "$latest_log"
