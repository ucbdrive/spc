#!/bin/bash
cd .. &&
python train_torcs.py \
    --save-path mpc_5_seg_cont_2 \
    --continuous \
    --use-seg \
    --num-total-act 2 \
    --pred-step 5 \
    --use-collision \
    --use-offroad \
    --use-distance \
    --sample-with-collision \
    --sample-with-offroad \
    --sample-with-distance \
    --num-same-step 1 \
    --data-parallel \
    --id 25 \
    --resume
