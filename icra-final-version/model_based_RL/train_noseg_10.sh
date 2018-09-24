#!/bin/bash
python train_torcs.py \
    --save-path mpc_model_based \
    --continuous \
    --num-total-act 2 \
    --pred-step 10 \
    --buffer-size 50000 \
    --epsilon-frames 100000 \
    --batch-size 32 \
    --use-collision \
    --use-offroad \
    --use-distance \
    --sample-with-collision \
    --sample-with-offroad \
    --num-same-step 1 \
    --data-parallel \
    --id -2 \
    --resume

