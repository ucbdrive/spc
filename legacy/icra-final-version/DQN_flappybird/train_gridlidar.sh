#!/bin/bash
for i in '00' '01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19'
do
    python train_dqn.py \
        --save-path 'gridlidar_'$i \
        --log-name 'log_lidar.txt' \
        --load-old-q-value \
        --seed 0 \
        --frame-history-len 1 \
        --env-id 'lidargrid' \
        --buffer-size 100000 \
        --device 'cuda' \
        --plan $i'.txt'
done
