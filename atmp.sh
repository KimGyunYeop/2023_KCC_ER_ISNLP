#/bin/bash
python -u main.py\
    --result_path "test" \
    --add_rnn_baseline \
    --dev

# python -u main.py\
#     --result_path "test" \
#     --image_transformer \
#     --ignore_bio \
#     --dev