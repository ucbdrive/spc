#!/bin/bash
cd ..
#while true; do
    python main.py \
        --env torcs \
        --learning-freq 10 \
        --num-train-steps 1 \
        --num-total-act 2 \
        --pred-step 10 \
        --buffer-size 50000 \
        --epsilon-frames 100000 \
        --batch-size 16 \
        --use-collision \
        --use-offroad \
        --use-speed \
        --sample-with-collision \
        --sample-with-offroad \
        --speed-threshold 20 \
        --use-guidance \
        --expert-bar 200 \
        --safe-length-collision 30 \
        --safe-length-offroad 15 \
        --data-parallel \
        --id 25 \
        --recording \
        --verbose \
        --resume
#done
