import logging
from time import time

import numpy as np
import pandas as pd

import torch
import os
import torch.nn as nn

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
import dataset.customdataset as customdataset
from dataset import CustomDataset
from hydra.utils import to_absolute_path
from .utils import set_seed, cal_kappa_score
from .optimizers import get_optimizer_grouped_parameters
from sklearn.metrics import cohen_kappa_score
from torchinfo import summary
import tqdm

from model import get_classifier, get_tree_classifier

from sklearn.model_selection import StratifiedKFold, train_test_split


logger = logging.getLogger(__name__)

class ExpBase:
    def __init__(self, config):
        set_seed(config.seed)
        self.seed = config.seed

        self.model_name = config.model.name
        self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        self.epochs = config.exp.epochs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.best_kappa = 0

        df: CustomDataset = getattr(customdataset, self.data_config.name)(seed=self.seed, **self.data_config)
        self.train, self.test = df.train, df.test
        self.feature_columns = df.feature_columns
        self.target_column = df.target_column
        self.train_dataset, self.val_dataset, self.train_loader, self.val_loader, self.test_loader = df.prepare_loaders()
        self.id = df.id
        self.num_labels = df.num_labels

        # Loss function
        self.loss_fn = CrossEntropyLoss().to(self.device)

    def encoder_train(self):

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.model.bert.encoder.layer.named_parameters():
            param.requires_grad = True
        
        for name, param in self.model.named_parameters():
            if param.requires_grad : 
                print(name)

    def train_epoch(self,n_examples):
        self.model.train()
        losses = []
        self.llm_pred_data = self.train[['essay_id', 'score']]

        for d in tqdm.tqdm(self.train_loader):
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["label"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )



            loss = self.loss_fn(outputs.logits, labels)
            losses.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        return np.mean(losses)
    
    def eval_model(self,n_examples):
        self.model.eval()
        losses = []
        true_labels = []
        pred_labels = []
        
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

                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
                losses.append(loss.item())

        kappa = cohen_kappa_score(true_labels,pred_labels,weights='quadratic')
        return kappa, np.mean(losses)
    
    def get_predictions(self):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for d in self.test_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                 
                _, preds = torch.max(outputs.logits, dim=1)

                preds += 1

                predictions.extend(preds)


        predictions = [pred.item() for pred in predictions]

        return predictions

    def run(self):
        model_config = self.get_model_config()
        self.model = get_classifier(
            self.model_name,
            device = self.device, 
            model_config = model_config, 
            num_labels=self.num_labels,
            seed=self.seed
        )
        # self.add_layer()

        if(self.model_name == "deberta"):
            self.model.resize_token_embeddings(len(self.train_dataset.tokenizer))

        # Optimizer and scheduler
        # optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.model, lr=2e-5,  weight_decay=0.01, lr_decay=0.95)
        # self.optimizer = AdamW(optimizer_grouped_parameters)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, correct_bias=False)
        self.total_steps = len(self.train_loader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=self.total_steps)

        logger.info(f"model name: {self.model_name} device: {self.device}")
        # self.encoder_train()

        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch + 1}/{self.epochs}')

            start_time = time()
            train_loss = self.train_epoch(len(self.train_dataset))
            logger.info(f'Train loss: {train_loss} train time: {time() - start_time}')

            start_time = time()
            val_kappa, val_loss = self.eval_model(len(self.val_dataset))
            logger.info(f'Val loss {val_loss} kappa {val_kappa} eval time: {time() - start_time}')
            
            if val_kappa >= self.best_kappa:
                self.best_kappa = val_kappa
                torch.save(self.model.state_dict(), 'best_model_state.bin')
            
        summary(self.model, depth=4)

        if os.path.exists('best_model_state.bin'):
            self.model.load_state_dict(torch.load('best_model_state.bin'))
        else:
            logger.error("No model file found. Ensure training completes successfully.")


        preds = self.get_predictions()

        submission_df = pd.DataFrame({
            'essay_id': self.id,
            'score': preds
        })
        print(submission_df)

        submission_df.to_csv('submission.csv', index=False)
    
    def get_model_config(self, *args, **kwargs):
        raise NotImplementedError()

    def add_layer(self):
        hidden_size = self.model.model.config.hidden_size if hasattr(self.model.model, 'config') else 768  # or any appropriate default

        # Check if the classifier exists and is linear
        if hasattr(self.model.model, 'classifier') and isinstance(self.model.model.classifier, nn.Linear):
            in_features = self.model.model.classifier.in_features
            out_features = self.model.model.classifier.out_features

            # New sequential classifier with additional layer
            self.model.model.classifier = nn.Sequential(
                nn.Linear(in_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_features)
            )
        else:
            raise AttributeError("The model does not have a compatible classifier to add a layer to.")

        self.model.to(self.device)
   
class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)
    
    def get_model_config(self, *args, **kwargs):
        return self.model_config

class ExpStacking(ExpBase):
    def __init__(self, config):
        super().__init__(config)

        self.input_dim = 6
        self.output_dim = 6
        self.n_splits = 10

    def each_fold(self, i_fold, train_data, val_data):
        x, y = self.get_x_y(train_data)

        model_config = self.get_model_config(i_fold=i_fold, x=x, y=y, val_data=val_data)
        model = get_tree_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
        )
        start = time()
        model.fit(
            x,
            y,
            eval_set=(val_data[self.feature_columns], val_data[self.target_column].values.squeeze()),
        )
        end = time() - start
        logger.info(f"[Fit {self.model_name}] Time: {end}")
        return model, end
    
    def run(self):
        self.train['score'] = self.train['score']-1
        

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        y_test_pred_all = []
        score_all = 0
        for i_fold, (train_idx, val_idx) in enumerate(skf.split(self.train, self.train[self.target_column])):

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)

            kappa = cal_kappa_score(model, val_data, self.feature_columns, self.target_column)

            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] kappa score: {kappa}"
            )

            score_all += kappa


            y_test_pred_all.append(
                model.predict_proba(self.test[self.feature_columns]).reshape(-1, 1, 6)
            )
        
        y_test_pred_all = np.argmax(np.concatenate(y_test_pred_all, axis=1).mean(axis=1), axis=1)
        submit_df = pd.DataFrame(self.id)
        submit_df['score'] = y_test_pred_all+1

        print(submit_df)
        submit_df.to_csv("submit.csv", index=False)

        logger.info(f" {self.model_name} score average: {score_all/self.n_splits} ")

    def get_model_config(self, *args, **kwargs):
            return self.model_config

    def get_x_y(self, train_data):
        x, y = train_data[self.feature_columns], train_data[self.target_column].values.squeeze()
        return x, y



