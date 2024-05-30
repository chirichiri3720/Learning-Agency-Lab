import logging
from time import time

import numpy as np
import pandas as pd

import torch

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from model import Bertmodel
from dataset import CustomDataset

logger = logging.getLogger(__name__)

class ExpBase:
    def __init__(self, config):
        self.epochs = config.epochs
        # self.model_name = config.model.name

        # self.model_config = config.model.params
        self.exp_config = config.exp
        # self.data_config = config.data

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Bertmodel(self.device, num_labels=6)

        self.best_accuracy = 0

        df = CustomDataset()
        self.train_dataset, self.val_dataset, self.train_loader, self.val_loader, self.test_loader = df.prepare_loaders()
        self.id = df.id

        # Optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.total_steps = len(self.train_loader) * 3
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)

        # Loss function
        self.loss_fn = CrossEntropyLoss().to(self.device)



    def train_epoch(self, n_examples):
        self.model.train()
        losses = []
        correct_predictions = 0
        for d in self.train_loader:
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["label"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        return correct_predictions.double() / n_examples, np.mean(losses)
    
    def eval_model(self, n_examples):
        self.model.eval()
        losses = []
        correct_predictions = 0
        with torch.no_grad():
            for d in self.val_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                _, preds = torch.max(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, labels)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)
    
    def get_predictions(self):
        self.model.eval()
        essay_ids = []
        predictions = []

        with torch.no_grad():
            for d in self.test_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.max(outputs.logits, dim=1)

                predictions.extend(preds)

            
        
        predictions = [pred.item() for pred in predictions]
        print(predictions)

        return predictions


    def run(self):
        logger.info(f"device: {self.device}")

        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch + 1}/{self.epochs}')

            train_acc, train_loss = self.train_epoch(len(self.train_dataset))

            logger.info(f'Train loss: {train_loss} accuracy: {train_acc}')

            val_acc, val_loss = self.eval_model(len(self.val_dataset))

            logger.info(f'Val loss {val_loss} accuracy {val_acc}')
            
            if val_acc >= self.best_accuracy:
                self.best_accuracy = val_acc
                torch.save(self.model.state_dict(), 'best_model_state.bin')
            
        
        self.model.load_state_dict(torch.load('best_model_state.bin'))

        preds = self.get_predictions()

        submission_df = pd.DataFrame({
            'essay_id': self.id,
            'score': preds
        })
        print(submission_df)

        submission_df.to_csv('submission.csv', index=False)
   
class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)
    

