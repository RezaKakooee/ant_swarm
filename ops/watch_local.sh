#!/usr/bin/env bash

if [ "${1:-}" = "--once" ]; then
    {
        for pid in $(ps -u "$USER" -o pid=); do
            job_name="$({ tr '\0' '\n' < "/proc/$pid/environ"; } 2>/dev/null |
                sed -nE 's/^(ANT_SWARM_RUN_ID|RADIAL_SPHERE_RUN_ID)=//p' | head -1)"
            if [ -n "$job_name" ]; then
                stats="$(ps -p "$pid" -o pid=,etimes=,%cpu=,%mem=,stat= 2>/dev/null)"
                [ -n "$stats" ] && echo "$job_name $stats"
            fi
        done
    } | awk '
        function elapsed(s) {
            d=int(s/86400); s%=86400; h=int(s/3600); s%=3600; m=int(s/60); s%=60
            return (d ? d "-" : "") sprintf("%02d:%02d:%02d", h, m, s)
        }
        {
            name=$1; pid=$2; sec=$3; cpu=$4; mem=$5; state=$6
            if (!(name in first_pid) || pid < first_pid[name]) first_pid[name]=pid
            if (sec > max_sec[name]) max_sec[name]=sec
            total_cpu[name]+=cpu; total_mem[name]+=mem
            if (!(name in job_state) || state ~ /^R/) job_state[name]=state
        }
        END {
            printf "%-8s %-12s %7s %7s %-5s %s\n", "PID", "ELAPSED", "CPU%", "MEM%", "STATE", "JOB_NAME"
            for (name in first_pid)
                printf "%-8s %-12s %7.1f %7.1f %-5s %s\n", first_pid[name], elapsed(max_sec[name]), total_cpu[name], total_mem[name], job_state[name], name
        }'
    exit
fi

watch -n "${1:-2}" "'$0' --once"
