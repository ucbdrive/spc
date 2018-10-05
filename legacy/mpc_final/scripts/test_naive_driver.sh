#!/bin/bash
cd .. &&
python test_naive_driver.py \
    --normalize \
    --seed 0 \
    --id 5 \
    --continuous \
    --use-collision \
    --use-offroad \
    --use-distance \
    --use-pos \
    --use-angle \
    --use-speed \
    --use-xyz \
    --sample-with-distance \
    --pred-step 15 \
    --use-dqn \
    --num-dqn-action 11
