import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from hydra.utils import to_absolute_path
from .dataset import EssayDataset

class CustomDataset():
    def __init__(self):
        super(CustomDataset, self).__init__()
        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = 512
        self.batch_size = 10

        # Load data
        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))

        # データ削減
        self.train = self.train[:100]

        self.feature_column = 'full_text'
        self.target_column = 'score'

        self.id = self.test['essay_id']
    
    def prepare_loaders(self):
        
        # Prepare datasets
        X_train, X_val, y_train, y_val = train_test_split(
            self.train[self.feature_column], self.train[self.target_column], test_size=0.1, random_state=42
        )
        train_dataset = EssayDataset(X_train.tolist(), y_train.tolist(), self.tokenizer, self.max_len)
        val_dataset = EssayDataset(X_val.tolist(), y_val.tolist(), self.tokenizer, self.max_len)
        test_dataset = EssayDataset(self.test[self.feature_column].tolist(), tokenizer=self.tokenizer, max_len=self.max_len)

        # Prepare dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_dataset, val_dataset, train_loader, val_loader, test_loader