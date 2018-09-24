#!/bin/bash
python train_torcs.py \
    --save-path mpc_10_cont_pong \
    --seed 6 \
    --eval \
    --recording \
    --learning-freq 10 \
    --num-train-steps 1 \
    --continuous \
    --one-hot \
    --use-seg \
    --lstm2 \
    --pred-step 4 \
    --buffer-size 50000 \
    --epsilon-frames 100000 \
    --batch-size 16 \
    --num-same-step 1 \
    --data-parallel \
    --id 25 \
    --resume