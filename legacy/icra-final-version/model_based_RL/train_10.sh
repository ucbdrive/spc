#!/bin/bash
/usr/bin/python3 train_torcs.py \
    --save-path mpc_model_based \
    --continuous \
    --use-seg \
    --one-hot \
    --lstm2 \
    --num-total-act 2 \
    --pred-step 10 \
    --buffer-size 50000 \
    --epsilon-frames 100000 \
    --batch-size 24 \
    --use-collision \
    --use-offroad \
    --use-distance \
    --sample-with-collision \
    --sample-with-offroad \
    --num-same-step 1 \
    --data-parallel \
    --id 19 \
    --resume

