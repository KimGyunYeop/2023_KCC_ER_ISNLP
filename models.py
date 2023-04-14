import torch
from torch import nn

from transformers import AutoModel, T5ForConditionalGeneration


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
            
            text_out.loss = loss_cur_turn + self.args.d * self.args.prev_turn * loss_prev_turn
            
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
            
            text_out.loss = loss_cur_turn + self.args.d * self.args.prev_turn * loss_prev_turn
            
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
    

class T5TextAudioAFModel(nn.Module):
    def __init__(self, args):
        super(T5TextAudioAFModel, self).__init__()
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
        if self.args.AF_num != 0:
            image_out = image_out[:,:voice_len - (voice_len % self.args.AF_num),:].reshape(b, self.args.AF_num, voice_len//self.args.AF_num, 768)
            image_out = torch.mean(image_out, dim=-2)
            
        # print(input_embeds.shape)
        # print(image_out.shape)
        input_embeds = torch.cat([input_embeds, image_out], dim=1)
        # print(input_embeds.shape)
        
        image_attention_mask = torch.ones(batch, self.args.AF_num).to(input_ids.device)
        attention_masks = torch.cat([attention_masks, image_attention_mask], dim=-1)
        
        # text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks, labels=label_input_ids)
        
        
        if self.args.prev_turn_loss:
            decoder_input_ids = self.text_encoder._shift_right(label_input_ids)
            text_out = self.text_encoder(attention_mask=attention_masks, inputs_embeds=input_embeds, decoder_input_ids=decoder_input_ids)
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss_cur_turn = loss_fct(text_out.logits[:,:3,:].reshape(-1, text_out.logits.size(-1)), label_input_ids[:,:3].reshape(-1))
            loss_prev_turn = loss_fct(text_out.logits[:,3:,:].reshape(-1, text_out.logits.size(-1)), label_input_ids[:,3:].reshape(-1))
            
            text_out.loss = loss_cur_turn + self.args.d * self.args.prev_turn * loss_prev_turn
            
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
        
        if self.args.AF_num != 0:
            image_out = image_out[:,:voice_len - (voice_len % self.args.AF_num),:].reshape(b, self.args.AF_num, voice_len//self.args.AF_num, 768)
            image_out = torch.mean(image_out, dim=-2)
            
        input_embeds = torch.cat([input_embeds, image_out], dim=1)
        # print(input_embeds.shape)
        
        image_attention_mask = torch.ones(batch, self.args.AF_num).to(input_ids.device)
        attention_masks = torch.cat([attention_masks, image_attention_mask], dim=-1)
        
        # text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_masks, labels=label_input_ids)
        text_out = self.text_encoder.generate(attention_mask=attention_masks, inputs_embeds=input_embeds)
        
        return text_out
    