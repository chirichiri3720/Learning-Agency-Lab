import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig

from transformers import AutoModelForSequenceClassification, AutoConfig



class Bertmodel(nn.Module):
    def __init__(self, device, num_labels=6, model_config=None):
        super(Bertmodel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.model = self.model.to(device)
        config = BertConfig(**model_config)
        self.model.config = config

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)

class Debertamodel(nn.Module):
    def __init__(self, device, num_labels=6, model_config=None):
        super(Debertamodel, self).__init__()

        config = AutoConfig.from_pretrained('microsoft/deberta-v3-base')
        config.num_labels = num_labels
        
        # モデル設定をカスタマイズ（必要に応じて）
        if model_config:
            for key, value in model_config.items():
                setattr(config, key, value)
        
        self.model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', config=config)
        self.model = self.model.to(device)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def resize_token_embeddings(self, new_vocab_size):
        self.model.resize_token_embeddings(new_vocab_size)
    

    
