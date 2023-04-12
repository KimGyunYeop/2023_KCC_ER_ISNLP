from customdatasets import T5TextAudioCumulDataset, TextAudioCumulDataset
from models import T5TextModel, T5TextAudioCAModel, T5TextAudioAEModel
from utils import parse_args, fix_seed

from transformers import AutoTokenizer, AutoFeatureExtractor
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

from sklearn.metrics import f1_score

# imageModel2imagePath = {
#     "wav2vec":"facebook/wav2vec2-base", 
#     "wavlm":"microsoft/wavlm-base", 
#     "whisper":"openai/whisper-base"
# }

fix_seed()
args = parse_args()
device = "cuda:"+str(args.gpu)
args.label = ['happy', 'fear', 'surprise', 'angry', 'neutral', 'sad', 'disqust']

def id2label_f(x):
    try:
        return args.label[x]
    except:
        return "None"
    
args.korean_label = ["기쁨", "두려움", "놀람", "분노", "중립", "슬픔", "논쟁"]
id2label = dict(zip(range(len(args.label)), args.label))

# args.audio_encoder_path = imageModel2imagePath[args.image_model.lower()]
print("model path : ", args.audio_encoder_path)
print(args)

args.result_path = os.path.join("results", args.result_path)
if args.dev:
    args.result_path="results/test"
else:
    if os.path.exists(args.result_path):
        print("already path exist!!")
        quit()

print(args.result_path)
os.makedirs(args.result_path, exist_ok=True)
os.makedirs(os.path.join(args.result_path,"pred"), exist_ok=True)
json.dump(vars(args), open(os.path.join(args.result_path,"config.json"), "w"), indent=2)

tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_path)
fe = AutoFeatureExtractor.from_pretrained(args.audio_encoder_path)
train_dataset = T5TextAudioCumulDataset(args, None, 30, tokenizer, fe)
eval_dataset = T5TextAudioCumulDataset(args, 30, None, tokenizer, fe)

if args.T5CAModel:
    model = T5TextAudioCAModel(args)
elif args.T5AEModel:
    model = T5TextAudioAEModel(args)
else:
    model = T5TextModel(args)
model.to(device)

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn)

optimizer = Adam(model.parameters(), lr=args.lr, eps=args.eps)

lf = nn.CrossEntropyLoss()
lf.to(device)

result_dict = {}
summary_dict = {}

summary_dict["max acc epoch"] = -1
summary_dict["max acc"] = 0
summary_dict["max macro_f1 epoch"] = -1
summary_dict["max macro_f1"] = 0
summary_dict["max micro_f1 epoch"] = -1
summary_dict["max micro_f1"] = 0
summary_dict["max weight_f1 epoch"] = -1
summary_dict["max weight_f1"] = 0

data_check = True

for E in range(args.epoch):
    model.train()
    train_loss_list = []
    all_train_pred = []
    all_train_label = []
    for data in tqdm(train_dataloader):
        optimizer.zero_grad()
        
        if data_check:
            for x in data:
                print("data check")
                print(x)
            data_check = False
        
        
        data = [x.to(device) if torch.is_tensor(x) else x for x in data]
                
        loss, pred = model(data[0], data[1], data[2], data[-2], data[-1])
        
        # if args.prev_turn_loss:
        #     pred = pred[:,:min(4,pred.size()[1]),:]
        #     data[-2] = data[-2][:,:min(4,data[-2].size()[1])]
        # loss = lf(pred, i[-1])
        # print(pred)
        label = tokenizer.batch_decode(data[-2])
        # print(label)
        if args.prev_turn_loss:
            label = list(map(lambda x:x.split("<extra_id_1>")[0], label))
        # print(label)
        
        label_list = []
        for li in range(data[0].size(0)):
            for j, tmp_label in enumerate(args.korean_label):
                if tmp_label in label[li]:
                    label_list.append(j)
                    break
        all_train_label.extend(label_list)
        # print(label_list)
        
        pred_label = tokenizer.batch_decode(torch.argmax(pred,dim=-1))
        # print(pred_label)
        if args.prev_turn_loss:
            pred_label = list(map(lambda x:x.split("<extra_id_1>")[0], pred_label))
        # print(pred_label)
        # print(pred_label)
        pred_list = []
        for li in range(data[0].size(0)):
            for j, tmp_label in enumerate(args.korean_label):
                if tmp_label in pred_label[li]:
                    pred_list.append(j)
                    break
                if j == (len(args.korean_label) - 1):
                    pred_list.append(-100)
        # print(pred_list)
        all_train_pred.extend(pred_list)
        loss.backward()
        
        
        optimizer.step()
        
        train_loss_list.append(loss.item())
        
        
    print(Counter(all_train_pred))
      
        
    train_loss = sum(train_loss_list)/len(train_loss_list)
    print("train loss in epoch {}: {}".format(str(E), train_loss))
    
    # print(all_train_pred)
    # print(all_train_label)
    train_acc = float(torch.sum(torch.Tensor(all_train_pred) == torch.Tensor(all_train_label)) / torch.Tensor(all_train_pred).size()[0])
    train_macro_f1 = f1_score(all_train_label, all_train_pred, average='macro')
    train_micro_f1 = f1_score(all_train_label, all_train_pred, average='micro')
    train_weight_f1 = f1_score(all_train_label, all_train_pred, average='weighted')
    # test_acc = sum(test_acc_list)/len(test_acc_list)
    print("train acc in epoch {}: {}".format(E, train_acc))
    print("train macro_f1 in epoch {}: {}".format(E, train_macro_f1))
    print("train micro_f1 in epoch {}: {}".format(E, train_micro_f1))
    print("train weight_f1 in epoch {}: {}".format(E, train_weight_f1))
    print("train label_count = ",Counter(all_train_label))
    print("train pred_count = ",Counter(all_train_pred))
        
    
    test_acc_list = []
    label_count = []
    pred_count = []
    result_df = []
    all_test_pred = []
    all_test_label = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(eval_dataloader):
            data = [x.to(device) if torch.is_tensor(x) else x for x in data]
            
            pred = model.test(data[0], data[1], data[2], data[-2], data[-1])
            
            
            label = tokenizer.batch_decode(data[-2])
            # print(label)
            if args.prev_turn_loss:
                label = list(map(lambda x:x.split("<extra_id_1>")[0], label))
            # print(label)
            
            label_list = []
            for i in range(data[0].size(0)):
                for j, tmp_label in enumerate(args.korean_label):
                    if tmp_label in label[i]:
                        label_list.append(j)
                        break
            all_test_label.extend(label_list)
            # print(label_list)
            
            pred_label = tokenizer.batch_decode(pred)
            # print(pred_label)
            if args.prev_turn_loss:
                pred_label = list(map(lambda x:x.split("<extra_id_1>")[0], pred_label))
            # print(pred_label)
            pred_list = []
            for i in range(data[0].size(0)):
                for j, tmp_label in enumerate(args.korean_label):
                    if tmp_label in pred_label[i]:
                        pred_list.append(j)
                        break
                    if j == (len(args.korean_label) - 1):
                        pred_list.append(-100)
            # print(pred_list)
            all_test_pred.extend(pred_list)
            # break
            # pred_label = torch.argmax(pred, dim=-1)
            # acc = torch.sum(torch.Tensor(pred_list) == label_list) / i[-1].size()[0]
            # label_count.extend(i[-1].reshape(-1).tolist())
            # pred_count.extend(pred_label.reshape(-1).tolist())
            
            # # print(pred_label)
            # # print(id2label)
            
            # i[-2]["pred"] = list(map(lambda x: id2label[x], pred_label.view(-1).tolist()))
            # i[-2]["correct"] = (pred_label == i[-1].view(-1)).tolist()
            
            # # print(acc)
            # test_acc_list.append(acc.item())
            result_df.append(data[-3])
            
    
    # print(Counter(all_test_pred))
    test_acc = float(torch.sum(torch.Tensor(all_test_pred) == torch.Tensor(all_test_label)) / torch.Tensor(all_test_pred).size()[0])
    test_macro_f1 = f1_score(all_test_label, all_test_pred, average='macro')
    test_micro_f1 = f1_score(all_test_label, all_test_pred, average='micro')
    test_weight_f1 = f1_score(all_test_label, all_test_pred, average='weighted')
    # test_acc = sum(test_acc_list)/len(test_acc_list)
    print("test acc in epoch {}: {}".format(E, test_acc))
    print("test macro_f1 in epoch {}: {}".format(E, test_macro_f1))
    print("test micro_f1 in epoch {}: {}".format(E, test_micro_f1))
    print("test weight_f1 in epoch {}: {}".format(E, test_weight_f1))
    print("label_count = ",Counter(all_test_label))
    print("pred_count = ",Counter(all_test_pred))
    
    result_dict[E] = {}
    result_dict[E]["train loss"] = train_loss
    result_dict[E]["train acc"] = train_acc
    result_dict[E]["train macro_f1"] = train_macro_f1
    result_dict[E]["train micro_f1"] = train_micro_f1
    result_dict[E]["train weight_f1"] = train_weight_f1
    result_dict[E]["test acc"] = test_acc
    result_dict[E]["test macro_f1"] = test_macro_f1
    result_dict[E]["test micro_f1"] = test_micro_f1
    result_dict[E]["test weight_f1"] = test_weight_f1
    result_dict[E]["label_count"] = Counter(all_test_label)
    result_dict[E]["pred_count"] = Counter(all_test_pred)
    
    if test_acc > summary_dict["max acc"]:
        summary_dict["max acc"] = test_acc
        summary_dict["max acc epoch"] = E
    if test_macro_f1 > summary_dict["max macro_f1"]:
        summary_dict["max macro_f1"] = test_macro_f1
        summary_dict["max macro_f1 epoch"] = E
    if test_micro_f1 > summary_dict["max micro_f1"]:
        summary_dict["max micro_f1"] = test_micro_f1
        summary_dict["max micro_f1 epoch"] = E
    if test_weight_f1 > summary_dict["max weight_f1"]:
        summary_dict["max weight_f1"] = test_weight_f1
        summary_dict["max weight_f1 epoch"] = E
    # print(result_dict)
    json.dump(result_dict, open(os.path.join(args.result_path, "result.json"), "w"), indent=2)
    
    result_df = pd.concat(result_df)
    result_df["pred"] = list(map(lambda x: id2label_f(x), all_test_pred))
    result_df["correct"] = (torch.Tensor(all_test_pred) == torch.Tensor(all_test_label)).tolist()
    result_df.to_csv(os.path.join(args.result_path,"pred", "predict_{}.csv".format(str(E))))

json.dump(summary_dict, open(os.path.join(args.result_path, "result_summary.json"), "w"), indent=2)
            
            

