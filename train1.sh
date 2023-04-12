#/bin/bash

PREV_TURN=(0 2)
CA_EMB_NUM=(0 1 10 100)
GPU=1

for prev_turn in ${PREV_TURN[@]}; do
    for ca_emb_num in ${CA_EMB_NUM[@]}; do

        python -u main.py\
            --result_path "t5_CA_prev_{$prev_turn}_embnum_{$ca_emb_num}" \
            --prev_turn $prev_turn \
            --T5CAModel \
            --CAEmbedding_num $ca_emb_num \
            --gpu $GPU

    done
done