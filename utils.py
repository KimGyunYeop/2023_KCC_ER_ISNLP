import argparse
from transformers.adapters import AutoAdapterModel

import random
import numpy as np
import torch


def fix_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="sentiment analysis")
    parser.add_argument(
        "--data_path", type=str, default="datasets",
        help="text encoder model path"
    )
    parser.add_argument(
        "--result_path", type=str, default="test",
        help="result_path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="batch_size"
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="batch_size"
    )
    parser.add_argument(
        "--text_encoder_path", type=str, default="paust/pko-t5-base",
        help="text_encoder_path"
    )
    parser.add_argument(
        "--audio_encoder_path", type=str, default="kresnik/wav2vec2-large-xlsr-korean",
        help="audio_encoder_path"
    )
    parser.add_argument(
        "--lr", type=float, default=0.00001,
        help="data path"
    )
    parser.add_argument(
        "--eps", type=float, default=0.001,
        help="data path"
    )
    parser.add_argument(
        "--epoch", type=int, default=30,
        help="epoch"
    )
    parser.add_argument(
        "--prev_turn", type=int, default=0,
        help="epoch"
    )
    parser.add_argument(
        "--T5CAModel", default=False, action='store_true',
        help="ignore_text"
    )
    parser.add_argument(
        "--CAEmbedding_num", type=int, default=0,
        help="epoch"
    )
    parser.add_argument(
        "--T5AEModel", default=False, action='store_true',
        help="ignore_text"
    )
    parser.add_argument(
        "--AEEmbedding_num", type=int, default=1,
        help="epoch"
    )
    parser.add_argument(
        "--prev_turn_loss", default=False, action='store_true',
        help="prev_turn_loss"
    )
    parser.add_argument(
        "--d", type=float, default=1.0,
        help="d"
    )
    parser.add_argument(
        "--dev", default=False, action='store_true',
        help="wave2vec"
    )
    args = parser.parse_args()
    return args