from customdatasets import TextAudioBioDataset
from utils import parse_args, fix_seed

from transformers import AutoTokenizer
from transformers import BertModel



fix_seed()
args = parse_args()

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

train_dataset = TextAudioBioDataset(args, None, 30, tokenizer)
eval_dataset = TextAudioBioDataset(args, 30, None, tokenizer)

train_dataset[0]
