#!/bin/bash
python3 train_torcs.py \
    --save-path mpc_5_cont_pretrain_coll \
    --continuous \
    --num-total-act 2 \
    --pred-step 5 \
    --buffer-size 50000 \
    --epsilon-frames 100000 \
    --batch-size 32 \
    --use-offroad \
    --use-collision \
    --use-distance \
    --sample-with-offroad \
    --sample-with-collision \
    --sample-with-distance \
    --num-same-step 1 \
    --id 25 \
    --resume
