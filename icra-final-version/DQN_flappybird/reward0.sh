#!/bin/bash
python train_dqn.py \
    --save-path 'gridworld-reward0' \
    --log-name 'log_train_gridworld0.txt' \
    --load-old-q-value \
    --seed 0 \
    --frame-history-len 1 \
    --env-id 'gridworld-v0' \
    --buffer-size 50000 \
    --batch-size 32 \
    --plan 'new_plan0.txt' \
    --device 'cuda' \
    --reward-mode 0
