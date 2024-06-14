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
    
    def add_layer(self,additional_layers=1):
        hidden_size = self.model.config.hidden_size
        in_features = self.model.classifier.in_features
        out_features = self.model.config.num_labels

        # 新しい分類器の構築
        classifier_layers = []
        for _ in range(additional_layers):
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, out_features))

        self.model.classifier = nn.Sequential(*classifier_layers)
        self.model.to(self.device)

class Debertamodel(nn.Module):
    def __init__(self, device, num_labels=6, model_config=None):
        super(Debertamodel, self).__init__()

        self.device = device
        config = AutoConfig.from_pretrained('microsoft/deberta-v3-base')
        config.num_labels = num_labels
        
        # モデル設定をカスタマイズ（必要に応じて）
        if model_config:
            for key, value in model_config.items():
                setattr(config, key, value)
        
        self.model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', config=config)
        self.model = self.model.to(device)
        self.classifier = self.model.classifier

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def resize_token_embeddings(self, new_vocab_size):
        self.model.resize_token_embeddings(new_vocab_size)
    
    def add_layer(self,additional_layers=1):
        # hidden_size = self.model.model.config.hidden_size if hasattr(self.model.model, 'config') else 768  # or any appropriate default
        hidden_size = self.model.config.hidden_size
        in_features = self.model.classifier.in_features
        out_features = self.model.config.num_labels

        # 新しい分類器の構築
        classifier_layers = []
        for _ in range(additional_layers):
            classifier_layers.append(nn.Linear(in_features, hidden_size))
            classifier_layers.append(nn.ReLU())
            in_features = hidden_size
        
        classifier_layers.append(nn.Linear(in_features, out_features))

        self.model.classifier = nn.Sequential(*classifier_layers)
        self.model.to(self.device)

    
