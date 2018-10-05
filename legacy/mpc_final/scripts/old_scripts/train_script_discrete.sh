#!/bin/bash
python3 train_torcs.py \
	--use-seg \
    --normalize \
    --num-total-act 2 \
	--use-collision \
	--use-offroad \
	--use-distance \
	--use-seg \
    --use-pos \
    --use-angle \
    --use-speed \
    --use-xyz