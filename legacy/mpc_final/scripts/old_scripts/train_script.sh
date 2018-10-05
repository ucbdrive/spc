#!/bin/bash
python2 train_torcs.py \
    --normalize \
    --use-dqn \
    --num-total-act 2 \
    --use-pos \
    --use-angle \
    --use-speed \
    --use-xyz 

