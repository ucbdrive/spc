#!/bin/bash
python3 train_torcs.py \
    --save-path mpc_5_withoutseg_cont \
    --continuous \
    --num-total-act 2 \
    --pred-step 5 \
    --use-collision \
    --use-offroad \
    --use-distance \
    --use-speed \
    --sample-with-collision \
    --sample-with-offroad \
    --sample-with-distance \
    --num-same-step 1 \
    --data-parallel \
    --id 8 \
    --resume
