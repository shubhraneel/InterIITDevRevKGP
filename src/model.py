import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

# Dual Encoder Model
class DualEncoder(pl.LightningModule):

    def __init__(self, vision_model_name, language_model_name, language_input_size = 768, 
                vision_hidden_size = 2048, output_size = 512, vision_learning_rate=1e-2, 
                language_learning_rate = 1e-5, dropout = 0.4, vision_pretrained = False, 
                language_pretrained = True, weight_decay=1e-4,
                warmup_epochs = 2):
        super().__init__()

        self.save_hyperparameters()

        self.vision_model_name = vision_model_name
        self.language_model_name = language_model_name
        self.language_input_size = language_input_size
        self.vision_hidden_size = vision_hidden_size
        self.output_size = output_size
        self.vision_learning_rate = vision_learning_rate
        self.language_learning_rate = language_learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.vision_pretrained = vision_pretrained
        self.language_pretrained = language_pretrained
        self.warmup_epochs = warmup_epochs

        self.loss_cls = ContrastiveLoss()

        self.vision_model = VisionModel(self.vision_model_name, hidden_size = self.vision_hidden_size, 
                                        output_size = self.output_size, pretrained = self.vision_pretrained)
        self.language_model = LanguageModel(self.language_model_name, input_size = language_input_size, 
                                            output_size = self.output_size, dropout = self.dropout)
    
        self.accuracy = torchmetrics.Accuracy()
    
    def on_epoch_start(self):
        print('\n')

    def forward(self, image, text_input_ids, attention_masks = None, token_type_ids = None):
        
        image_features = self.vision_model(image)
        text_features = self.language_model(text_input_ids, attention_masks = attention_masks, token_type_ids = token_type_ids)

        return image_features, text_features
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        
        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])
        
        self.vision_scheduler.step()
        self.language_scheduler.step()

        return loss
    
    def validation_step(self, batch, batch_idx):

        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])

        return loss
    
    def test_step(self, batch, batch_idx):

        image_features, text_features = self.forward(batch['images'], batch['caption_input_ids'], batch['caption_attention_masks'], batch['caption_token_type_ids'])
        loss = self.loss_cls(image_features, text_features, batch['image_ids'])

        return loss
    
    def configure_optimizers(self):

        vision_optimizer = optim.Adam(self.vision_model.parameters(), lr=self.hparams.vision_learning_rate, weight_decay=self.weight_decay)
        language_optimizer = optim.Adam(self.language_model.parameters(), lr=self.hparams.language_learning_rate, weight_decay=self.weight_decay)

        self.vision_scheduler = CosineAnnealingLR(vision_optimizer, T_max = self.warmup_epochs)
        self.language_scheduler = CosineAnnealingLR(language_optimizer, T_max = self.warmup_epochs)

        return [vision_optimizer, language_optimizer]