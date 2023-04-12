import torch
from torch import nn

from transformers import AutoModel, T5ForConditionalGeneration

class TextAudioBioModel(nn.Module):
    def __init__(self, args):
        super(TextAudioBioModel, self).__init__()
        self.args = args
        
        self.text_encoder = AutoModel.from_pretrained(args.text_encoder_path)
        self.text_head = nn.Linear(768, len(args.label))
        
        if args.image_transformer:
            self.wav2vec_model = AutoModel.from_pretrained(args.audio_encoder_path)
            self.projector = nn.Linear(768, 512)
            self.classifier = nn.Linear(512, len(args.label))
        else:
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
        
        self.bio_classificaiton_haed = nn.Linear(32, len(args.label))
        
        
    def forward(self, input_ids, attention_masks, mel_image, bio_feature):
        result_prob = []
        
        if input_ids is not None:
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks)
            text_out = self.text_head(text_out.pooler_output)
            
            result_prob.append(text_out)
            
        if mel_image is not None:
            if self.args.image_transformer:
                # print(self.wav2vec_model(mel_image))
                mel_out = self.wav2vec_model(mel_image).last_hidden_state
                # print(mel_out.shape)
                mel_out = self.projector(mel_out).mean(dim=1)
                # print(mel_out.shape)
                mel_out = self.classifier(mel_out)
                result_prob.append(mel_out)
                # print(mel_out.shape)
            else:
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
            
        # print(result_prob)
        result_prob = sum(result_prob)
        # print(result_prob)
        # print(result_prob.shape)
        
        return result_prob
    

class TextAudioRNNModel(nn.Module):
    def __init__(self, args):
        super(TextAudioRNNModel, self).__init__()
        self.args = args
        
        self.text_encoder = AutoModel.from_pretrained(args.text_encoder_path)
        self.wav2vec_model = AutoModel.from_pretrained(args.audio_encoder_path)
        self.projector = nn.Linear(768, 768)
        
        self.rnn = nn.RNN(768*2, 768, batch_first=True)
        self.classifier = nn.Linear(768, len(args.label))
        
        
    def forward(self, input_ids, attention_masks, images, tmp=None):
        # print(input_ids.shape)
        # print(attention_masks.shape)
        # print(images.shape)
        
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks).pooler_output
        
        # print(text_out.shape)
            
        # print(self.wav2vec_model(mel_image))
        image_out = self.wav2vec_model(images).last_hidden_state
        # print(mel_out.shape)
        image_out = self.projector(image_out).mean(dim=1)
            
        # print(image_out.shape)
        
        out = torch.cat([text_out, image_out], dim=-1)
        
        out = out.reshape(-1, self.args.prev_turn + 1, 768*2)
        # print(out.shape)
        
        out = self.rnn(out)[0]
        # print(out.shape)
        out = self.classifier(out[:,-1,:])
        # print(out.shape)
            
        return out
    

class T5TextModel(nn.Module):
    def __init__(self, args):
        super(T5TextModel, self).__init__()
        self.args = args
        
        self.text_encoder = T5ForConditionalGeneration.from_pretrained(args.text_encoder_path)
        self.wav2vec_model = AutoModel.from_pretrained(args.audio_encoder_path)
        self.projector = nn.Linear(768, 768)
        
        self.rnn = nn.RNN(768*2, 768, batch_first=True)
        self.classifier = nn.Linear(768, len(args.label))
        
        
    def forward(self, input_ids, attention_masks, images, label_input_ids, label_attention_mask):
        # print(input_ids.shape)
        # print(attention_masks.shape)
        # print(images.shape)
        if self.args.prev_turn_loss:
            decoder_input_ids = self.text_encoder._shift_right(label_input_ids)
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks, decoder_input_ids=decoder_input_ids)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_cur_turn = loss_fct(text_out.logits[:,:3,:].reshape(-1, text_out.logits.size(-1)), label_input_ids[:,:3].reshape(-1))
            loss_prev_turn = loss_fct(text_out.logits[:,3:,:].reshape(-1, text_out.logits.size(-1)), label_input_ids[:,3:].reshape(-1))
            
            text_out.loss = loss_cur_turn + self.args.d * loss_prev_turn
            
        else:
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks, labels=label_input_ids)
        # print(text_out.loss)
            
        return text_out.loss, text_out.logits
    
    def test(self, input_ids, attention_masks, images, label_input_ids, label_attention_mask):
        
        text_out = self.text_encoder.generate(input_ids=input_ids, attention_mask=attention_masks)
        # print(text_out)
        
        return text_out
    
    
class T5TextAudioCAModel(nn.Module):
    def __init__(self, args):
        super(T5TextAudioCAModel, self).__init__()
        self.args = args
        
        self.text_model = T5ForConditionalGeneration.from_pretrained(args.text_encoder_path)
        self.text_encoder = self.text_model.encoder
        
        self.wav2vec_model = AutoModel.from_pretrained(args.audio_encoder_path)
        self.projector = nn.Linear(1024, 768)
        
        self.rnn = nn.RNN(768*2, 768, batch_first=True)
        self.classifier = nn.Linear(768, len(args.label))
        
        
    def forward(self, input_ids, attention_masks, images, label_input_ids, label_attention_mask):
        # print(input_ids.shape)
        # print(attention_masks.shape)
        # print(images.shape)
        
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks)
        # print(text_out[0].shape)
        image_out = self.wav2vec_model(images)[0]
        image_out = self.projector(image_out)
        
        b, voice_len, _ = image_out.size()
        
        if self.args.CAEmbedding_num != 0:
            image_out = image_out[:,:voice_len - (voice_len % self.args.CAEmbedding_num),:].reshape(b, self.args.CAEmbedding_num, voice_len//self.args.CAEmbedding_num, 768)
            image_out = torch.mean(image_out, dim=-2)
            
        # print(image_out.shape)
        # print(torch.cat([text_out[0], image_out], dim=1).shape)
            
        text_out.last_hidden_state = torch.cat([text_out[0], image_out], dim=1)
        
        
        if self.args.prev_turn_loss:
            decoder_input_ids = self.text_model._shift_right(label_input_ids)
            text_out = self.text_model(encoder_outputs=text_out, decoder_input_ids=decoder_input_ids)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_cur_turn = loss_fct(text_out.logits[:,:3,:].reshape(-1, text_out.logits.size(-1)), label_input_ids[:,:3].reshape(-1))
            loss_prev_turn = loss_fct(text_out.logits[:,3:,:].reshape(-1, text_out.logits.size(-1)), label_input_ids[:,3:].reshape(-1))
            
            text_out.loss = loss_cur_turn + self.args.d * loss_prev_turn
            
        else:
            text_out = self.text_model(encoder_outputs=text_out, labels=label_input_ids)
            
        # text_out = self.text_model(encoder_outputs=text_out, labels=label_input_ids)
        # print(text_out.loss)
        
            
        return text_out.loss, text_out.logits
    
    def test(self, input_ids, attention_masks, images, label_input_ids, label_attention_mask):
        
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks)
        # print(text_out[0].shape)
        image_out = self.wav2vec_model(images)[0]
        image_out = self.projector(image_out)
        # print(image_out.shape)
        
        b, voice_len, _ = image_out.size()
        
        if self.args.CAEmbedding_num != 0:
            image_out = image_out[:,:voice_len - (voice_len % self.args.CAEmbedding_num),:].reshape(b, self.args.CAEmbedding_num, voice_len//self.args.CAEmbedding_num, 768)
            image_out = torch.mean(image_out, dim=-2)
            
        # print(text_out[0])
        # print(image_out)
        text_out.last_hidden_state = torch.cat([text_out[0], image_out], dim=1)
        # text_out[0] = encoder_output
        # print(text_out[0].shape)
        
        text_out = self.text_model.generate(encoder_outputs=text_out)
        # print(text_out)
        
        return text_out
    

class T5TextAudioAEModel(nn.Module):
    def __init__(self, args):
        super(T5TextAudioAEModel, self).__init__()
        self.args = args
        
        self.text_encoder = T5ForConditionalGeneration.from_pretrained(args.text_encoder_path)
        self.text_encoder_emb = self.text_encoder.shared # self.text_encoder.shared
        
        self.wav2vec_model = AutoModel.from_pretrained(args.audio_encoder_path)
        self.projector = nn.Linear(1024, 768)
        
        
    def forward(self, input_ids, attention_masks, images, label_input_ids, label_attention_mask):
        # print(input_ids.shape) # torch.Size([16, 134])
        # print(attention_masks.shape)
        # print(images.shape) # torch.Size([16, 284416])
        
        input_embeds = self.text_encoder_emb(input_ids)
        # print(input_embeds.shape) # torch.Size([16, 134, 768])
        
        batch = input_embeds.size()[0]

        # print(self.wav2vec_model(mel_image))
        image_out = self.wav2vec_model(images).last_hidden_state
        image_out = self.projector(image_out)
        # print(mel_out.shape)
        
        b, voice_len, _ = image_out.size()
        
        # print(image_out.shape)
        if self.args.AEEmbedding_num != 0:
            image_out = image_out[:,:voice_len - (voice_len % self.args.AEEmbedding_num),:].reshape(b, self.args.AEEmbedding_num, voice_len//self.args.AEEmbedding_num, 768)
            image_out = torch.mean(image_out, dim=-2)
            
        # print(input_embeds.shape)
        # print(image_out.shape)
        input_embeds = torch.cat([input_embeds, image_out], dim=1)
        # print(input_embeds.shape)
        
        image_attention_mask = torch.ones(batch, self.args.AEEmbedding_num).to(input_ids.device)
        attention_masks = torch.cat([attention_masks, image_attention_mask], dim=-1)
        
        # text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks, labels=label_input_ids)
        
        
        if self.args.prev_turn_loss:
            decoder_input_ids = self.text_encoder._shift_right(label_input_ids)
            text_out = self.text_encoder(attention_mask=attention_masks, inputs_embeds=input_embeds, decoder_input_ids=decoder_input_ids)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_cur_turn = loss_fct(text_out.logits[:,:3,:].reshape(-1, text_out.logits.size(-1)), label_input_ids[:,:3].reshape(-1))
            loss_prev_turn = loss_fct(text_out.logits[:,3:,:].reshape(-1, text_out.logits.size(-1)), label_input_ids[:,3:].reshape(-1))
            
            text_out.loss = loss_cur_turn + self.args.d * loss_prev_turn
            
        else:
            text_out = self.text_encoder(attention_mask=attention_masks, inputs_embeds=input_embeds, labels=label_input_ids)
            
        # text_out = self.text_encoder(attention_mask=attention_masks, inputs_embeds=input_embeds, labels=label_input_ids)
        
        return text_out.loss, text_out.logits
    
    def test(self, input_ids, attention_masks, images, label_input_ids, label_attention_mask):
        # print(input_ids.shape) # torch.Size([16, 134])
        # print(attention_masks.shape)
        # print(images.shape) # torch.Size([16, 284416])
        
        input_embeds = self.text_encoder_emb(input_ids)
        # print(input_embeds.shape) # torch.Size([16, 134, 768])
        
        batch = input_embeds.size()[0]

        # print(self.wav2vec_model(mel_image))
        image_out = self.wav2vec_model(images).last_hidden_state
        image_out = self.projector(image_out)
        # print(mel_out.shape)
        b, voice_len, _ = image_out.size()
        
        if self.args.AEEmbedding_num != 0:
            image_out = image_out[:,:voice_len - (voice_len % self.args.AEEmbedding_num),:].reshape(b, self.args.AEEmbedding_num, voice_len//self.args.AEEmbedding_num, 768)
            image_out = torch.mean(image_out, dim=-2)
            
        input_embeds = torch.cat([input_embeds, image_out], dim=1)
        # print(input_embeds.shape)
        
        image_attention_mask = torch.ones(batch, self.args.AEEmbedding_num).to(input_ids.device)
        attention_masks = torch.cat([attention_masks, image_attention_mask], dim=-1)
        
        # text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks, labels=label_input_ids)
        text_out = self.text_encoder.generate(attention_mask=attention_masks, inputs_embeds=input_embeds)
        
        return text_out
    