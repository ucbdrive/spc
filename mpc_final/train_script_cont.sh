#!/bin/bash
python3.5 train_torcs.py \
	--resume \
	--continuous \
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
    --use-xyz \
    --use-dqn