#!/usr/bin/env bash
srun --job-name=rob_srun \
     --partition=performance \
     --time=06:00:00 \
     --nodes=1 \
     --ntasks-per-node=1 \
     --cpus-per-task=20 \
     --mem=120G \
     --gres=gpu:1 \
     --export=ALL \
     --pty bash --init-file /home2/reza/sim/.sim_init.sh