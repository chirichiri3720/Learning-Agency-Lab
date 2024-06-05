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
    def __init__(
        self,
        seed: int = 42,
        max_len: int = 512,
        batch_size: int = 2,
        test_size: float = 0.2,
        feature_column: str = 'full_text',
        target_column: str = 'score',
        num_labels: int = 6,
        **kwargs
    ):
        # Load data
        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.seed = seed
        self.max_len = max_len
        self.batch_size =  batch_size
        self.test_size = test_size

        # Load data
        self.train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        self.test = pd.read_csv(to_absolute_path("datasets/test.csv"))

        # データ削減
        self.train = self.train[:10]

        self.feature_column = feature_column
        self.target_column = target_column
        self.num_labels = num_labels

        self.id = self.test['essay_id']
    
    def prepare_loaders(self):
        # Prepare datasets
        X_train, X_val, y_train, y_val = train_test_split(
            self.train[self.feature_column], self.train[self.target_column], test_size=self.test_size, random_state=self.seed
        )
        train_dataset = EssayDataset(X_train.tolist(), y_train.tolist(), self.tokenizer, self.max_len)
        val_dataset = EssayDataset(X_val.tolist(), y_val.tolist(), self.tokenizer, self.max_len)
        test_dataset = EssayDataset(self.test[self.feature_column].tolist(), tokenizer=self.tokenizer, max_len=self.max_len)

        # Prepare dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        
        return train_dataset, val_dataset, train_loader, val_loader, test_loader

class V0(CustomDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
