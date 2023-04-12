#/bin/bash

PREV_TURN=(0 1 2)
GPU=0

for prev_turn in ${PREV_TURN[@]}; do
    python -u main.py \
        --result_path "t5_baseline_prev_$prev_turn" \
        --prev_turn $prev_turn \
        --gpu $GPU
done