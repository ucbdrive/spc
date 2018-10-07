#!/bin/bash
/usr/bin/python3 train_torcs.py \
    --save-path mpc_10_cont_nopretrain \
    --env carla \
    --simple-seg \
    --continuous \
    --one-hot \
    --use-seg \
    --lstm2 \
    --num-total-act 2 \
    --pred-step 10 \
    --buffer-size 50000 \
    --epsilon-frames 100000 \
    --batch-size 24 \
    --use-collision \
    --use-offroad \
    --use-speed \
    --sample-with-collision \
    --sample-with-offroad \
    --sample-with-speed \
    --num-same-step 1 \
    --data-parallel \
    --id 25 \
    --resume