#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2 python train_torcs.py \
    --save-path mpc_12_fb \
    --learning-freq 100 \
    --num-train-steps 10 \
    --continuous \
    --one-hot \
    --use-seg \
    --lstm2 \
    --pred-step 12 \
    --buffer-size 50000 \
    --epsilon-frames 50000 \
    --batch-size 32 \
    --num-same-step 1 \
    --data-parallel \
    --id 25 \
    --resume
