from torch import nn

from transformers import AutoModel

class TextAudioBioModel(nn.Module):
    def __init__(self, args):
        super(TextAudioBioModel, self).__init__()
        
        self.text_encoder = AutoModel.from_pretrained(args.text_encoder_path)
        self.text_head = nn.Linear(768, len(args.label))
        
        self.cnn1 = nn.Conv2d(1, 32, 2, 2)
        self.cnn2 = nn.Conv2d(32, 64, 2)
        self.cnn3 = nn.Conv2d(64, 128, 2)
        
        self.cnn_list = [self.cnn1, self.cnn2, self.cnn3]
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.image_haed1 = nn.Linear(17920, 64)
        self.image_haed2 = nn.Linear(64, 32)
        self.image_haed3 = nn.Linear(32, len(args.label))
        self.image_haed_list = [self.image_haed1, self.image_haed2, self.image_haed3]
        
        self.bio_haed1 = nn.Linear(4, 256)
        self.bio_haed2 = nn.Linear(256, 128)
        self.bio_haed3 = nn.Linear(128, 64)
        self.bio_haed4 = nn.Linear(64, 32)
        self.bio_haed_list = [self.bio_haed1, self.bio_haed2, self.bio_haed3, self.bio_haed4]
        
        self.bio_classificaiton_haed = nn.Linear(64, len(args.label))
        
        
    def forward(self, input_ids, attention_masks, mel_image, bio_feature):
        result_prob = []
        
        if input_ids is not None:
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks)
            text_out = self.text_head(text_out.pooler_output)
            
            result_prob.append(text_out)
            
        if mel_image is not None:
            batch_size = mel_image.size()[0]
            mel_out = mel_image
            for i in self.cnn_list:
                mel_out = self.pool(i(mel_out))
                mel_out = self.relu(mel_out)
                
            mel_out = mel_out.view(batch_size, -1)
            
            for i in self.image_haed_list:
                mel_out = i(mel_out)
                mel_out = self.relu(mel_out)
            
            result_prob.append(mel_out)
            
        if bio_feature is not None:
            bio_out = bio_feature
            for i in self.bio_haed_list:
                bio_out = i(bio_out)
                bio_out = self.relu(bio_out)
            
            bio_out = self.bio_classificaiton_haed(bio_out)
            result_prob.append(bio_out)
            
        result_prob = sum(result_prob)
        
        return result_prob