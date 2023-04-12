#/bin/bash
# python -u main.py\
#     --result_path "t5_baseline" \
#     --gpu 0 \
#     --prev_turn_loss \
#     --prev_turn 2 \
#     --batch_size 32 \
#     --d 0.5 \
#     --dev 


python -u main.py \
    --result_path "t5_baseline_prev_{2}_prev_loss_0.3" \
    --prev_turn 2 \
    --prev_turn_loss \
    --d 0.3 \
    --gpu 3

# python -u main.py\
#     --result_path "test" \
#     --image_transformer \
#     --ignore_bio \
#     --dev