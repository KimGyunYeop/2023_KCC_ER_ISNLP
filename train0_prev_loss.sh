#/bin/bash


PREV_TURN=(0 2)
GPU=0

for prev_turn in ${PREV_TURN[@]}; do
    python -u main.py \
        --result_path "t5_baseline_prev_$prev_turn" \
        --prev_turn $prev_turn \
        --gpu $GPU
done

PREV_TURN=(2)
D=(0.7 0.3 1)
GPU=0

for prev_turn in ${PREV_TURN[@]}; do
    for d in ${D[@]}; do
        python -u main.py \
            --result_path "t5_baseline_prev_{$prev_turn}_prev_loss_$d" \
            --prev_turn $prev_turn \
            --prev_turn_loss \
            --d $d \
            --gpu $GPU
    done
done