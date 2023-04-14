import os
import re
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset

import torchaudio
import torchaudio.transforms as AT

from collections import Counter


class T5TextAudioCumulDataset(Dataset):
    def __init__(self, args, start_index, end_index, tokenizer, wav2vec_fe=None) -> None:
        super(T5TextAudioCumulDataset, self).__init__()
        self.args = args
        self.label = args.label
        self.label2id = dict(zip(self.label, range(len(self.label))))
        self.id2label = dict(zip(range(len(self.label)), self.label))
        self.id2klabel = dict(zip(range(len(self.label)), self.args.korean_label))
        
        annotation_path = os.path.join(args.data_path, "annotation")
        self.wav_path = os.path.join(args.data_path, "wav")
        self.session_list = list(map(lambda x : re.sub(r'[^0-9]', '', x), os.listdir(annotation_path)))
        
        if start_index == None:
            start_index = 0
        
        if end_index == None:
            self.session_list = self.session_list[start_index:]
        else:
            self.session_list = self.session_list[start_index:end_index]
        print(self.session_list)
        self.tokenizer = tokenizer
        self.wav2vec_fe =wav2vec_fe
        self.defalut_columns=["turn","WAV_start","WAV_end","Segment ID","Total Evaluation","Valence","Arousal", "session_id", "text"]
        self.annotations = []
        
        
        for i in self.session_list:
            data = pd.read_csv(os.path.join(annotation_path,"Sess{}_eval.csv".format(i)))
            data = data.iloc[1:,:7]
            data["session_id"] = i
            # data["text"] = data["session_id"]
            data["text"] = data["Segment ID"].apply(lambda x: open(os.path.join(self.wav_path, "Session{}".format(i),"{}.txt".format(x)), encoding="cp949").read()[:-1]) #-1은 마지막 \n 삭제
            
            data.columns = self.defalut_columns
            self.annotations.append(data)
        
            # print(data)
            
            
        self.annotations = pd.concat(self.annotations).reset_index(drop=True)
        self.annotations["pred"]=""
        self.annotations["correct"]=""
        print(self.annotations)
        print(len(self.annotations))
        print(Counter(self.annotations["Total Evaluation"]))
        
        
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        index_data_annotation = self.annotations.iloc[index,:]
        
        # print(data_annotation)
        # print(set(";".join(list(self.annotations["Total Evaluation"])).split(";")))
        text = index_data_annotation["text"]
        # print(text)
        session_id = index_data_annotation["session_id"]
        segment_id = index_data_annotation["Segment ID"]
        turn = index_data_annotation["turn"]
        final_label = self.label2id[index_data_annotation["Total Evaluation"].split(";")[0]]
        
        texts = []
        # audios = []
        labels = []
        data_annotations = []
        
        # session_id = data_annotation["session_id"]
        # segment_id = data_annotation["Segment ID"]
        audio = torchaudio.load(os.path.join(self.wav_path, "Session{}".format(session_id),"{}.wav".format(segment_id)))
        # audios.append(audio)
        # print(index)
        # print((0, int(turn), - self.args.prev_turn))
        # print(data_annotation)
        for e, i in enumerate(range(int(turn), int(turn)-self.args.prev_turn-1 , -1)):
            # if i <= 0:
            #     text = ""
            #     audio = (torch.Tensor([[0, 0]]), 16000)
            #     data_annotation = None
            #     label = -100
            # else:            
            data_annotation = self.annotations.iloc[index - (self.args.prev_turn - e),:]
            # print(index - e)
            # print(data_annotation)
            
            # print(data_annotation)
            # print(set(";".join(list(self.annotations["Total Evaluation"])).split(";")))
            text = data_annotation["text"]
            
            # print(text)
            
            label = self.label2id[data_annotation["Total Evaluation"].split(";")[0]]
                
            data_annotations.append(data_annotation)
            texts.append(text)
            labels.append(label)
        
        if not self.args.prev_turn_loss:
            labels = final_label
        # print(texts)
        # print(labels)
        
        return texts, audio, index_data_annotation, labels
    
    def collate_fn(self, batchs):
        # print(batchs)
        #print(batchs[0])
        
        if self.args.prev_turn_loss:
            texts = []
            # audios = []
            for i in batchs:
                # print(i[0])
                text = ""
                for j in range(len(i[0])-1):
                    text = text + i[0][j] + " 현재 발화의 감정은 <extra_id_{}>, ".format(j+1)
                
                text = "<s> " + text + i[0][-1] + " </s> 마지막 발화자의 감정은 <extra_id_0>"
                texts.append(text)
        
        else:
            texts = []
            # audios = []
            for i in batchs:
                # print(i[0])
                text = ", ".join(i[0])
                text = "<s> " + text + " </s> 마지막 발화자의 감정은 <extra_id_0>"
                texts.append(text)
                # texts.extend(i[0])
                # for j in i[1]:
                #     audios.append(j[0].squeeze().tolist())
                #     # print(j[0])
        
        #  print(texts)
        # print(audios)
        tokens = self.tokenizer(text=texts, padding="longest", return_tensors="pt", max_length=256, truncation="longest_first")
        input_ids=tokens.input_ids
        attention_mask=tokens.attention_mask
        # print(input_ids.shape)
        # print(attention_mask.shape)
        
        # for i in audios:
        #     print(i.shape)
        # print(batchs[0])
        # print(batchs[0][1])
        audios = [batch[1][0].squeeze().tolist() for batch in batchs]
        # print(audios.shape)
        audios = self.wav2vec_fe(audios, sampling_rate = 16000, return_tensors="pt", padding="longest", max_length=100000, truncation="longest_first")
        audios = audios["input_values"]
        # print(audios.shape)
        
        
        # data = pd.concat([batch[-2] for batch in batchs],axis=1).T
        data = pd.concat([batch[-2] for batch in batchs],axis=1).T
        
        
        if self.args.prev_turn_loss:
            label_tmp = [batch[-1] for batch in batchs]
            labels = []
            for i in label_tmp:
                # print(i[0])
                label = ""
                for j in range(len(i)-1):
                    label = label + "<extra_id_{}>".format(j+1) + self.id2klabel[i[j]] + " "
                
                label = "<extra_id_0>" + self.id2klabel[i[-1]] + " " + label
                labels.append(label)
            
        else:
            labels = [batch[-1] for batch in batchs]
            labels = list(map(lambda x: "<extra_id_0> " + self.id2klabel[x], labels))
        # print(labels)
        labels = self.tokenizer(text=labels, padding="longest", return_tensors="pt", max_length=256, truncation="longest_first")
        
        # print(labels.input_ids.shape)
        # for i in range(self.args.batch_size):
        #     print(labels.input_ids[i,:])
        #     print(self.tokenizer.convert_ids_to_tokens(labels.input_ids[i,:]))
            
        # label = ""
        # for i in self.id2klabel.values():
        #     label = label + "<extra_id_0>" + i + " "
        # print(label)
        # print(self.tokenizer(text=label, padding="longest", return_tensors="pt", max_length=256, truncation="longest_first").input_ids)
        # print(self.tokenizer.convert_ids_to_tokens(self.tokenizer(text=label, padding="longest", return_tensors="pt", max_length=256, truncation="longest_first").input_ids[0,:]))
            
        
        # quit()
        # print(labels.input_ids.shape)
        # print(labels)
        # print(labels.input_ids)
        # print(self.tokenizer.convert_ids_to_tokens(labels.input_ids[0,:].tolist()))
        
        return input_ids, attention_mask, audios, data, labels.input_ids, labels.attention_mask
