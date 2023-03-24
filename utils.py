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
        "--result_path", type=str, default=None,
        help="text encoder model path"
    )
    args = parser.parse_args()
    return args