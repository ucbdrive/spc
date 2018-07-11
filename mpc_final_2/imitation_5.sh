#!/bin/bash
python3 train_torcs.py \
    --save-path imitation_5_seg_cont \
    --continuous \
    --imitation \
    --use-seg \
    --num-total-act 2 \
    --pred-step 5 \
    --use-collision \
    --use-offroad \
    --use-distance \
    --sample-with-collision \
    --sample-with-offroad \
    --sample-with-distance \
    --num-same-step 1 \
    --batch-size 8 \
    --save-freq 2 \
    --id 25 \
    --resume
