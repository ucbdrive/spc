#!/bin/bash
cd ..
python train_torcs.py \
    --lr 0.0003 \
    --env 'torcs-v0' \
    --save-path 'mpc_12_step' \
    --batch-size 7 \
    --pred-step 12 \
    --normalize \
    --buffer-size 30000 \
    --save-freq 10 \
    --learning-starts 100 \
    --learning-freq 50 \
    --target-update-freq 10 \
    --epsilon-frames 30000 \
    --frame-history-len 3 \
    --num-total-act 6 \
    --batch-step 500
