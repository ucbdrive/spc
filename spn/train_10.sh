#!/bin/bash
python train_torcs.py \
    --save-path mpc_10_cont_nopretrain \
    --continuous \
    --num-total-act 2 \
    --pred-step 10 \
    --use-collision \
    --batch-size 32 \
    --use-offroad \
    --use-distance \
    --sample-with-distance \
    --num-same-step 1 \
    --data-parallel \
    --id -15 \
    --pretrained False \
    --num-train-step 15 \
    --resume
