#!/bin/bash
python train_torcs.py \
    --save-path mpc_1_noseg_cont \
    --continuous \
    --num-total-act 2 \
    --pred-step 1 \
    --use-collision \
    --use-offroad \
    --use-distance \
    --sample-with-distance \
    --num-same-step 1 \
    --data-parallel \
    --no-supervise \
    --id 1 \
    --resume
