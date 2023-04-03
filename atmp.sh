#/bin/bash
python -u main.py\
    --result_path "test" \
    --ignore_bio \
    --image_transformer \
    --image_model Whisper \
    --dev \
    --batch_size 4