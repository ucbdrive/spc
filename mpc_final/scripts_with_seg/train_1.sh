#!/bin/bash
python3 train_torcs.py \
    --lr 0.001 \
    --use-seg \
    --frame-history-len 3 \
    --pred-step 1 \
    --batch-size 32 \
    --save-path 'cont_pred_1_seg_dqn_10_loss_dist' \
    --epsilon-frames 40000 \
    --target-update-freq 100 \
    --data-parallel \
    --id 0 \
    --continuous \
    --use-distance \
    --num-dqn-action 10 \
    --sample-with-distance
