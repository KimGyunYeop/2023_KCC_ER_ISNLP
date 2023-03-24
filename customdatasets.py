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

class TextAudioBioDataset(Dataset):
    def __init__(self, args, start_index, end_index, tokenizer) -> None:
        super(TextAudioBioDataset, self).__init__()
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
        print(self.annotations)
        print(len(self.annotations))
        
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
        audio, sr = torchaudio.load(os.path.join(self.wav_path, "Session{}".format(data_annotation["session_id"]),"{}.wav".format(data_annotation["Segment ID"])))
        # print(audio.shape)
        # print(sr)
        
        # plt.plot(audio[0,:])
        # plt.savefig("tmp.png")
        
        mel_audio = AT.MFCC(sample_rate=sr)(audio)
        # print(mel_audio.shape)
        mel_audio = self.resize_image(mel_audio)
        # print(mel_audio)
        # print(mel_audio.shape)
        
        # plt.imsave("mel_tmp.png",to_pil_image(mel_audio))
        
        label = self.label2id[data_annotation["Total Evaluation"].split(";")[0]]
        
        return text, mel_audio, None, label
    
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
            mel_audio=None
        else:
            mel_audio = torch.stack([batch[1] for batch in batchs], dim=0)
        
        if self.args.ignore_bio:
            bio_feature=None
        else:
            bio_feature = None
        
        
        labels = torch.LongTensor([batch[-1] for batch in batchs])
        
        return input_ids, attention_mask, mel_audio, bio_feature, labels