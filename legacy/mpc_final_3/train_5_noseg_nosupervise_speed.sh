#!/bin/bash
python train_torcs.py \
    --save-path mpc_5_noseg_nosupervise_speed \
    --continuous \
    --num-total-act 2 \
    --pred-step 5 \
    --use-speed \
    --target-speed 25 \
    --num-same-step 1 \
    --no-supervise \
    --data-parallel \
    --id 27 \
    --resume
