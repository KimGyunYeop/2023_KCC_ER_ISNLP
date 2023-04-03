#/bin/bash

# python -u main.py\
#     --result_path "wav2vec_text_audio_bio" \
#     --image_transformer

python -u main.py\
    --result_path "wavLM_text_audio" \
    --ignore_bio \
    --image_transformer \
    --image_model wavLM \
    --gpu 1 \
    --batch_size 2

python -u main.py\
    --result_path "wavLM_audio" \
    --ignore_text \
    --ignore_bio \
    --image_transformer \
    --image_model wavLM \
    --gpu 1 \
    --batch_size 2

# python -u main.py\
#     --result_path "baseline_text" \
#     --ignore_audio \
#     --ignore_bio \
#     --image_transformer

# python -u main.py\
#     --result_path "baseline_bio" \
#     --ignore_text \
#     --ignore_audio  \
#     --image_transformer

# python -u main.py\
#     --result_path "baseline_text_bio" \
#     --ignore_audio  \
#     --image_transformer
    
# python -u main.py\
#     --result_path "wav2vec_audio_bio" \
#     --ignore_text  \
#     --image_transformer