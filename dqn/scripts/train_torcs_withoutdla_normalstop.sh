#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=1,0 python run_dqn_atari.py \
    --use-cuda \
    --save-path 'dqn_torcs_withoutdla_normalstop' \
    --env-id 'torcs-v0' \
    --log-name 'log_train_dqn_torcs.txt' \
    --buffer-size 30000 \
    --seed 0 \
    --batch-size 32 \
    --train-dqn \
    --learning-rate 0.0001 \
    --learning-starts 100 \
    --learning-freq 10 \
    --target-update-freq 100 \
    --epsilon-frames 30000 \
    --optimizer 'adam' \
    --config 'quickrace_discrete_single.xml' \
    --without-dla \
    --num-total-act 6
