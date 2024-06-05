import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertConfig



class Bertmodel(nn.Module):
    def __init__(self, device, num_labels=6, model_config=None):
        super(Bertmodel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.model = self.model.to(device)
        config = BertConfig(**model_config)
        self.model.config = config

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    

    
