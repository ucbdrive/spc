#!/bin/bash
cd .. &&
python train_torcs.py \
    --normalize \
    --seed 0 \
    --id 8 \
    --continuous \
    --save-freq 100 \
    --resume \
    --use-collision \
    --use-offroad \
    --use-distance \
    --use-pos \
    --use-angle \
    --use-speed \
    --use-xyz \
    --sample-with-distance \
    --pred-step 1 \
    --use-dqn \
    --num-dqn-action 11 \
    --save-path 'cont_norandreset_noseg_dqn11_pred1_loss_dist' \
