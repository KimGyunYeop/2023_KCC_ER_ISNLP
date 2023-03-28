#/bin/bash

# python -u main.py\
#     --result_path "wav2vec_text_audio_bio" \
#     --wav2vec

python -u main.py\
    --result_path "wav2vec_text_audio" \
    --ignore_bio \
    --wav2vec \
    --gpu 1

python -u main.py\
    --result_path "wav2vec_audio" \
    --ignore_text \
    --ignore_bio \
    --wav2vec \
    --gpu 1

# python -u main.py\
#     --result_path "baseline_text" \
#     --ignore_audio \
#     --ignore_bio \
#     --wav2vec

# python -u main.py\
#     --result_path "baseline_bio" \
#     --ignore_text \
#     --ignore_audio  \
#     --wav2vec

# python -u main.py\
#     --result_path "baseline_text_bio" \
#     --ignore_audio  \
#     --wav2vec
    
# python -u main.py\
#     --result_path "wav2vec_audio_bio" \
#     --ignore_text  \
#     --wav2vec