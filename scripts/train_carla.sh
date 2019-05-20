#!/bin/bash
cd ..
#while true; do
    python main.py \
        --env carla \
        --learning-freq 100 \
        --num-train-steps 10 \
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
        --speed-threshold 10 \
        --use-guidance \
        --expert-bar 200 \
        --safe-length-collision 50 \
        --safe-length-offroad 30 \
        --data-parallel \
        --id 25 \
        --recording \
        --verbose \
        --resume
#done
