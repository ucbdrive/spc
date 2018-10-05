#!/bin/bash
cd ..&&
python train_torcs.py \
    --lr 0.001 \
    --frame-history-len 3 \
    --pred-step 10 \
    --batch-size 32 \
    --save-path 'cont_pred_10_loss_dist' \
    --epsilon-frames 40000 \
    --target-update-freq 100 \
    --data-parallel \
    --id -55 \
    --continuous \
    --use-collision \
    --use-distance \
    --use-offroad \
    --use-angle \
    --use-speed \
    --use-xyz \
    --sample-with-offroad \
    --sample-with-collision \
    --sample-with-distance \
    --num-train-steps 10 \
    --resume 
