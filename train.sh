#/bin/bash

python -u main.py\
    --result_path "baseline_text_audio_bio"

python -u main.py\
    --result_path "baseline_text_audio" \
    --ignore_bio
    
python -u main.py\
    --result_path "baseline_audio" \
    --ignore_text \
    --ignore_bio
    
python -u main.py\
    --result_path "baseline_text" \
    --ignore_audio \
    --ignore_bio