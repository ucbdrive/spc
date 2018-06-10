#!/bin/bash
python3 train_torcs.py \
    --save-path mpc_10_seg_cont \
    --continuous \
    --use-seg \
    --num-total-act 2 \
    --pred-step 10 \
    --use-speed \
    --use-seg \
    --num-same-step 1 \
    --data-parallel \
    --target-speed 25 \
    --batch-size 12 \
    --id 0 \
    --resume
