#/bin/bash

PREV_TURN=(1)
D=(0.7)
GPU=1

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


PREV_TURN=(2)
CA_EMB_NUM=(10)
D=0.7

for prev_turn in ${PREV_TURN[@]}; do
    for ca_emb_num in ${CA_EMB_NUM[@]}; do
        python -u main.py\
            --result_path "T5AEModel_prev_{$prev_turn}_embnum_{$ca_emb_num}_prevloss_$D" \
            --prev_turn $prev_turn \
            --prev_turn_loss \
            --d $D \
            --T5AEModel \
            --AEEmbedding_num $ca_emb_num \
            --gpu $GPU \
            --batch_size 6
    done 
done

# PREV_TURN=(2)
# CA_EMB_NUM=(50 10)
# GPU=0

# for prev_turn in ${PREV_TURN[@]}; do
#     for ca_emb_num in ${CA_EMB_NUM[@]}; do
#         python -u main.py\
#             --result_path "T5AEModel_prev_{$prev_turn}_embnum_{$ca_emb_num}_prevloss_0.3" \
#             --prev_turn $prev_turn \
#             --prev_turn_loss \
#             --d 0.3 \
#             --T5AEModel \
#             --AEEmbedding_num $ca_emb_num \
#             --gpu $GPU \
#             --batch_size 6
#     done 
# done


