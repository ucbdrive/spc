#!/bin/bash
cd ..&&
python train_torcs.py \
    --lr 0.001 \
    --frame-history-len 3 \
    --pred-step 15 \
    --batch-size 32 \
    --save-path 'cont_pred_15_dqn_10_loss_dist' \
    --epsilon-frames 40000 \
    --target-update-freq 100 \
    --data-parallel \
    --id 0 \
    --continuous \
    --use-collision \
    --use-offroad \
    --use-distance \
    --use-pos \
    --use-angle \
    --use-speed \
    --use-xyz \
    --use-dqn \
    --num-dqn-action 10 \
    --sample-with-collision \
    --sample-with-distance 
