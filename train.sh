#/bin/bash

python -u main.py\
    --result_path "baseline_text_audio"
    
python -u main.py\
    --result_path "baseline_audio"\
    --ignore_text
    
python -u main.py\
    --result_path "baseline_text"\
    --ignore_audio