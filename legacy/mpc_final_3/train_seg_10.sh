#!/bin/bash
python train_torcs.py \
    --save-path mpc_10_noseg_cont \
    --continuous \
    --use-seg \
    --num-total-act 2 \
    --pred-step 10 \
    --use-collision \
    --use-offroad \
    --use-distance \
    --sample-with-distance \
    --num-same-step 1 \
    --data-parallel \
    --batch-size 12 \
    --no-supervise \
    --id 10 \
    --resume
