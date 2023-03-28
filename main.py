from customdatasets import TextAudioBioDataset
from models import TextAudioBioModel
from utils import parse_args, fix_seed

from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from transformers import BertModel

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch

from tqdm import tqdm

import pandas as pd
import json
import os

from collections import Counter

fix_seed()
args = parse_args()
device = "cuda:"+str(args.gpu)
args.label = ['happy', 'fear', 'surprise', 'angry', 'neutral', 'sad', 'disqust']
id2label = dict(zip(range(len(args.label)), args.label))

args.result_path = os.path.join("results", args.result_path)
if args.dev:
    args.result_path="test"
else:
    if os.path.exists(args.result_path):
        print("already path exist!!")
        quit()

os.makedirs(args.result_path, exist_ok=True)
os.makedirs(os.path.join(args.result_path,"pred"), exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)
fe=None
if args.wav2vec:
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.audio_encoder_path)

train_dataset = TextAudioBioDataset(args, None, 30, tokenizer, fe)
eval_dataset = TextAudioBioDataset(args, 30, None, tokenizer, fe)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

model = TextAudioBioModel(args)
model.to(device)

optimizer = Adam(model.parameters(), lr=args.lr, eps=args.eps)

lf = nn.CrossEntropyLoss()
lf.to(device)

result_dict = {}
summary_dict = {}

summary_dict["max epoch"] = -1
summary_dict["max acc"] = 0

data_check = True

for E in range(args.epoch):
    model.train()
    train_loss_list = []
    for i in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        if data_check:
            for x in i:
                print("data check")
                print(x)
            data_check = False
        
        
        i = [x.to(device) if torch.is_tensor(x) else x for x in i]
                
        pred = model(i[0], i[1], i[2], i[3])
        loss = lf(pred, i[-1])
        loss.backward()
        
        
        optimizer.step()
        
        train_loss_list.append(loss.item())
        
    train_loss = sum(train_loss_list)/len(train_loss_list)
    print("train loss in epoch {}: {}".format(str(E), train_loss))
        
    
    test_acc_list = []
    label_count = []
    pred_count = []
    result_df = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(eval_dataloader):
            i = [x.to(device) if torch.is_tensor(x) else x for x in i]
            
            pred = model(i[0], i[1], i[2], i[3])
            
            pred_label = torch.argmax(pred, dim=-1)
            acc = torch.sum(pred_label == i[-1].view(-1)) / i[-1].size()[0]
            label_count.extend(i[-1].reshape(-1).tolist())
            pred_count.extend(pred_label.reshape(-1).tolist())
            
            i[-2]["pred"] = list(map(lambda x: id2label[x], pred_label.reshape(-1).tolist()))
            i[-2]["correct"] = (pred_label == i[-1].view(-1)).tolist()
            
            # print(acc)
            test_acc_list.append(acc.item())
            result_df.append(i[-2])
        
    test_acc = sum(test_acc_list)/len(test_acc_list)
    print("test acc in epoch {}: {}".format(E, test_acc))
    print("label_count = ",Counter(label_count))
    print("pred_count = ",Counter(pred_count))
    
    result_dict[E] = {}
    result_dict[E]["train loss"] = train_loss
    result_dict[E]["test acc"] = test_acc
    result_dict[E]["label_count"] = Counter(label_count)
    result_dict[E]["pred_count"] = Counter(pred_count)
    
    if test_acc > summary_dict["max acc"]:
        summary_dict["max acc"] = test_acc
        summary_dict["max epoch"] = E
    
    json.dump(result_dict, open(os.path.join(args.result_path, "result.json"), "w"), indent=2)
    
    result_df = pd.concat(result_df)
    result_df.to_csv(os.path.join(args.result_path,"pred", "predict_{}.csv".format(str(E))))

json.dump(summary_dict, open(os.path.join(args.result_path, "result_summary.json"), "w"), indent=2)
            
            

