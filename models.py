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
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        
        self.image_haed1 = nn.Linear(17920, 64)
        self.image_haed2 = nn.Linear(64, 32)
        self.image_haed3 = nn.Linear(32, len(args.label))
        
        
    def forward(self, input_ids, attention_masks, mel_image, bio_feature):
        result_prob = []
        
        if input_ids is not None:
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks)
            text_out = self.text_head(text_out.pooler_output)
            
            result_prob.append(text_out)
            
        if mel_image is not None:
            batch_size = mel_image.size()[0]
            mel_out = self.pool(self.cnn1(mel_image))
            mel_out = self.relu(mel_out)
            mel_out = self.pool(self.cnn2(mel_out))
            mel_out = self.relu(mel_out)
            mel_out = self.pool(self.cnn3(mel_out))
            mel_out = self.relu(mel_out)
            mel_out = mel_out.view(batch_size, -1)
            mel_out = self.image_haed1(mel_out)
            mel_out = self.relu(mel_out)
            mel_out = self.image_haed2(mel_out)
            mel_out = self.relu(mel_out)
            mel_out = self.image_haed3(mel_out)
            mel_out = self.relu(mel_out)
            
            result_prob.append(mel_out)
            
        result_prob = sum(result_prob)
        
        return result_prob