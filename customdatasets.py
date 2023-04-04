import os
import re
import pandas as pd
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset

import torchaudio
import torchaudio.transforms as AT

from torchvision.transforms.functional import to_pil_image
from torchvision import transforms as VT

from collections import Counter

class TextAudioBioDataset(Dataset):
    def __init__(self, args, start_index, end_index, tokenizer, wav2vec_fe=None) -> None:
        super(TextAudioBioDataset, self).__init__()
        self.args = args
        self.label = args.label
        self.label2id = dict(zip(self.label, range(len(self.label))))
        
        annotation_path = os.path.join(args.data_path, "annotation")
        self.wav_path = os.path.join(args.data_path, "wav")
        self.eda_path = os.path.join(args.data_path, "EDA")
        self.ibi_path = os.path.join(args.data_path, "IBI")
        self.temp_path = os.path.join(args.data_path, "TEMP")
        self.session_list = list(map(lambda x : re.sub(r'[^0-9]', '', x), os.listdir(annotation_path)))
        
        if start_index == None:
            start_index = 0
        
        if end_index == None:
            self.session_list = self.session_list[start_index:]
        else:
            self.session_list = self.session_list[start_index:end_index]
            
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
        
        if not args.ignore_bio:
            self.eda_dict = {}
            for i in os.listdir(self.eda_path):
                for j in os.listdir(os.path.join(self.eda_path, i)):
                    eda_data = pd.read_csv(os.path.join(self.eda_path, i, j), names=["value", "time", "segid"]).dropna()
                    for k in set(eda_data["segid"]):
                        self.eda_dict[i] = list(eda_data.loc[eda_data["segid"]==k, "value"])
            
            # print(self.eda_dict)
            self.ibi1_dict = {}
            self.ibi2_dict = {}
            for i in os.listdir(self.ibi_path):
                for j in os.listdir(os.path.join(self.ibi_path, i)):
                    ibi_data = pd.read_csv(os.path.join(self.ibi_path, i, j), names=["value1","value2", "time", "segid"]).dropna()
                    for k in set(ibi_data["segid"]):
                        self.ibi1_dict[i] = list(ibi_data.loc[ibi_data["segid"]==k, "value1"])
                        self.ibi2_dict[i] = list(ibi_data.loc[ibi_data["segid"]==k, "value2"])
            
            # print(self.temp_dict)
            self.temp_dict = {}
            for i in os.listdir(self.temp_path):
                for j in os.listdir(os.path.join(self.temp_path, i)):
                    temp_data = pd.read_csv(os.path.join(self.temp_path, i, j), names=["value", "time", "segid"]).dropna()
                    for k in set(temp_data["segid"]):
                        self.temp_dict[i] = list(temp_data.loc[temp_data["segid"]==k, "value"])
        
        # print(self.eda_dict)
                    
        self.resize_image = VT.Resize((100,465))
        
    
    # def load_text(self, session_id):
    #     with open()
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        data_annotation = self.annotations.iloc[index,:]
        
        # print(data_annotation)
        # print(set(";".join(list(self.annotations["Total Evaluation"])).split(";")))
        text = data_annotation["text"]
        # print(text)
        session_id = data_annotation["session_id"]
        segment_id = data_annotation["Segment ID"]
        audio = torchaudio.load(os.path.join(self.wav_path, "Session{}".format(session_id),"{}.wav".format(segment_id)))
        # print(audio.shape)
        # print(sr)
        
        # plt.plot(audio[0,:])
        # plt.savefig("tmp.png")
        
        if not self.args.image_transformer:
            audio = AT.MFCC(sample_rate=audio[1])(audio[0])
            # print(mel_audio.shape)
            audio = self.resize_image(audio)
        # print(mel_audio)
        # print(mel_audio.shape)
        
        # plt.imsave("mel_tmp.png",to_pil_image(mel_audio))
        
        label = self.label2id[data_annotation["Total Evaluation"].split(";")[0]]
        
        if self.args.ignore_bio:
            bio_feature = None
        else:
            bio_feature = []
            for i in [self.eda_dict, self.ibi1_dict, self.ibi2_dict, self.temp_dict]:
                try:
                    bio_feature.append(sum(i[segment_id])/len(i[segment_id]))
                except KeyError:
                    # i[segment_id]
                    bio_feature.append(0)
        
        
        return text, audio, bio_feature, data_annotation, label
    
    def collate_fn(self, batchs):
        if self.args.ignore_text:
            input_ids=None
            attention_mask=None
        else:
            texts = [batch[0] for batch in batchs]
            tokens = self.tokenizer(text=texts, padding="longest", return_tensors="pt", max_length=256, truncation="longest_first")
            input_ids=tokens.input_ids
            attention_mask=tokens.attention_mask
        
        if self.args.ignore_audio:
            audios=None
        else:
            if self.args.image_transformer:
                # print(batchs[0][1][1])
                # print([batch[1][1] for batch in batchs])
                # print([batch[1][0] for batch in batchs])
                # for i in [batch[1][0] for batch in batchs]:
                #     print(i.shape)
                
                if self.args.image_model.lower() == "whisper":
                    audios = self.wav2vec_fe([batch[1][0].squeeze().tolist() for batch in batchs], sampling_rate = batchs[0][1][1], return_tensors="pt")
                    audios = audios.input_features
                else:
                    audios = self.wav2vec_fe([batch[1][0].squeeze().tolist() for batch in batchs], sampling_rate = batchs[0][1][1], return_tensors="pt", padding="longest")
                    audios = audios["input_values"]

                # print(audios)
                # print(audios.shape)
            else:
                audios = torch.stack([batch[1] for batch in batchs], dim=0)
        
        if self.args.ignore_bio:
            bio_feature=None
        else:
            bio_feature = torch.Tensor([batch[2] for batch in batchs])
        
        data = pd.concat([batch[-2] for batch in batchs],axis=1).T
        labels = torch.LongTensor([batch[-1] for batch in batchs])
        
        return input_ids, attention_mask, audios, bio_feature, data, labels
    
class TextAudioCumulDataset(Dataset):
    def __init__(self, args, start_index, end_index, tokenizer, wav2vec_fe=None) -> None:
        super(TextAudioCumulDataset, self).__init__()
        self.args = args
        self.label = args.label
        self.label2id = dict(zip(self.label, range(len(self.label))))
        
        annotation_path = os.path.join(args.data_path, "annotation")
        self.wav_path = os.path.join(args.data_path, "wav")
        self.session_list = list(map(lambda x : re.sub(r'[^0-9]', '', x), os.listdir(annotation_path)))
        
        if start_index == None:
            start_index = 0
        
        if end_index == None:
            self.session_list = self.session_list[start_index:]
        else:
            self.session_list = self.session_list[start_index:end_index]
            
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
        
        self.resize_image = VT.Resize((100,465))
        
    
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
        audios = []
        labels = []
        data_annotations = []
        
        
        # print(index)
        # print((0, int(turn), - self.args.prev_turn))
        # print(data_annotation)
        for e, i in enumerate(range(int(turn), int(turn)-self.args.prev_turn-1 , -1)):
            if i <= 0:
                text = ""
                audio = (torch.Tensor([[0, 0]]), 16000)
                data_annotation = None
                label = -100
            else:
                data_annotation = self.annotations.iloc[index - e,:]
                # print(index - e)
                # print(data_annotation)
                
                # print(data_annotation)
                # print(set(";".join(list(self.annotations["Total Evaluation"])).split(";")))
                text = data_annotation["text"]
                
                # print(text)
                session_id = data_annotation["session_id"]
                segment_id = data_annotation["Segment ID"]
                audio = torchaudio.load(os.path.join(self.wav_path, "Session{}".format(session_id),"{}.wav".format(segment_id)))
                
                label = self.label2id[data_annotation["Total Evaluation"].split(";")[0]]
                
            data_annotations.append(data_annotation)
            texts.append(text)
            audios.append(audio)
            labels.append(label)
        
        labels = final_label
        
        return texts, audios, index_data_annotation, labels
    
    def collate_fn(self, batchs):
        # print(batchs)
        
        texts = []
        audios = []
        for i in batchs:
            texts.extend(i[0])
            for j in i[1]:
                audios.append(j[0].squeeze().tolist())
                # print(j[0])
        
        # print(texts)
        # print(audios)
        tokens = self.tokenizer(text=texts, padding="longest", return_tensors="pt", max_length=256, truncation="longest_first")
        input_ids=tokens.input_ids
        attention_mask=tokens.attention_mask
        # print(input_ids.shape)
        # print(attention_mask.shape)
        
        # for i in audios:
        #     print(i.shape)
        audios = self.wav2vec_fe(audios, sampling_rate = 16000, return_tensors="pt", padding="longest")
        audios = audios["input_values"]
        # print(audios.shape)
        
        
        # data = pd.concat([batch[-2] for batch in batchs],axis=1).T
        data = pd.concat([batch[-2] for batch in batchs],axis=1).T
        labels = torch.LongTensor([batch[-1] for batch in batchs]).reshape(-1)
        # print(labels.shape)
        
        
        return input_ids, attention_mask, audios, data, labels