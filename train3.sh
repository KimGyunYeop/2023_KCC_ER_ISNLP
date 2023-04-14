#/bin/bash

# PREV_TURN=(1)
# D=(1)
# GPU=3

# for prev_turn in ${PREV_TURN[@]}; do
#     for d in ${D[@]}; do
#         python -u main.py \
#             --result_path "t5_baseline_prev_{$prev_turn}_prev_loss_$d" \
#             --prev_turn $prev_turn \
#             --prev_turn_loss \
#             --d $d \
#             --gpu $GPU
#     done
# done

PREV_TURN=(0)
CA_EMB_NUM=(1 10 50)
GPU=3

for prev_turn in ${PREV_TURN[@]}; do
    for ca_emb_num in ${CA_EMB_NUM[@]}; do
        python -u main.py\
            --result_path "T5AEModel_prev_{$prev_turn}_embnum_{$ca_emb_num}" \
            --prev_turn $prev_turn \
            --T5AEModel \
            --AEEmbedding_num $ca_emb_num \
            --gpu $GPU
    done 
done