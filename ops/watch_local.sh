#!/usr/bin/env bash

if [ "${1:-}" = "--once" ]; then
    printf "%-8s %-10s %7s %7s %-6s %s\n" "PID" "ELAPSED" "CPU%" "MEM%" "STATE" "COMMAND / CONFIG"
    echo "--------------------------------------------------------------------------------"
    ps -u "$USER" -o pid,etimes,%cpu,%mem,stat,args | grep -E "python3.*(train_|evaluate|il_|sac_|generate_|scripts/)" | grep -v grep | while read -r pid etimes cpu mem stat args; do
        d=$((etimes / 86400)); s=$((etimes % 86400)); h=$((s / 3600)); s=$((s % 3600)); m=$((s / 60)); sec=$((s % 60))
        if [ $d -gt 0 ]; then
            el=$(printf "%d-%02d:%02d:%02d" $d $h $m $sec)
        else
            el=$(printf "%02d:%02d:%02d" $h $m $sec)
        fi
        cmd=$(echo "$args" | sed -E 's|.*/python3 -u? ||' | sed -E 's|.*/python3 ||')
        printf "%-8s %-10s %7.1f %7.1f %-6s %s\n" "$pid" "$el" "$cpu" "$mem" "$stat" "$cmd"
    done
    exit 0
fi

watch -n "${1:-2}" "$0 --once"
