#!/bin/bash
python3 train_torcs.py \
    --save-path mpc_5_cont_pretrain_new \
    --continuous \
    --num-total-act 2 \
    --pred-step 5 \
    --buffer-size 50000 \
    --epsilon-frames 100000 \
    --batch-size 32 \
    --use-angle \
    --use-pos \
    --sample-with-angle \
    --sample-with-pos \
    --use-distance \
    --sample-with-distance \
    --num-same-step 1 \
    --data-parallel \
    --id 25 \
    --resume
