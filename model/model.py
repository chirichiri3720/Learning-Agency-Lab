import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup



class Bertmodel(nn.Module):
    def __init__(self, device, num_labels=6):
        super(Bertmodel, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
        self.model = self.model.to(device)
        self.model.config.hidden_dropout_prob = 0.2
        self.model.config.attention_probs_dropout_prob = 0.2
        self.model.config.output_attentions = False
        self.model.config.output_hidden_states = False

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask)
    

    
