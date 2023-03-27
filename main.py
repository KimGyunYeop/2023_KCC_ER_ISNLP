from customdatasets import TextAudioBioDataset
from models import TextAudioBioModel
from utils import parse_args, fix_seed

from transformers import AutoTokenizer
from transformers import BertModel

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import torch

from tqdm import tqdm

import json
import os

fix_seed()
args = parse_args()
device = "cuda:"+str(args.gpu)
args.label = ['happy', 'fear', 'surprise', 'angry', 'neutral', 'sad', 'disqust']

args.result_path = os.path.join("results", args.result_path)
if os.path.exists(args.result_path):
    print("already path exist!!")
    quit()

os.makedirs(args.result_path, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)

train_dataset = TextAudioBioDataset(args, None, 30, tokenizer)
eval_dataset = TextAudioBioDataset(args, 30, None, tokenizer)

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

for E in range(args.epoch):
    model.train()
    train_loss_list = []
    for i in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        # for x in i:
        #     print(x)
        #     print(x.shape)
        
        
        i = [x.to(device) if x is not None else None for x in i]
                
        pred = model(i[0], i[1], i[2], i[3])
        loss = lf(pred, i[-1])
        loss.backward()
        
        
        optimizer.step()
        
        train_loss_list.append(loss.item())
        
    train_loss = sum(train_loss_list)/len(train_loss_list)
    print("train loss in epoch {}: {}".format(str(E), train_loss))
        
    
    test_acc_list = []
    with torch.no_grad():
        model.eval()
        for i in tqdm(eval_dataloader):
            i = [x.to(device) if x is not None else None for x in i]
            pred = model(i[0], i[1], i[2], i[3])
        
            acc = torch.sum((torch.argmax(pred, dim=-1) == i[-1].view(-1))) / i[-1].size()[0]
            
            # print(acc)
            test_acc_list.append(acc.item())
        
    test_acc = sum(test_acc_list)/len(test_acc_list)
    print("test acc in epoch {}: {}".format(E, test_acc))
    
    result_dict[E] = {}
    result_dict[E]["train loss"] = train_loss
    result_dict[E]["test acc"] = test_acc
    
    if test_acc > summary_dict["max acc"]:
        summary_dict["max acc"] = test_acc
        summary_dict["max epoch"] = E
    
    json.dump(result_dict, open(os.path.join("result.json", "w")))

json.dump(summary_dict, open(os.path.join("result_summary.json", "w")))
            
            

