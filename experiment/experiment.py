import numpy as np
import pandas as pd

from hydra.utils import to_absolute_path

class ExpBase:
    def __init__(self, config):
        self.config = config
    
    def run(self):
        raise NotImplementedError

class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)

    def run(self):
        train = pd.read_csv(to_absolute_path("datasets/train.csv"))
        test = pd.read_csv(to_absolute_path("datasets/test.csv"))
        
        texts = train['full_text'].tolist()
        labels = train['score'].tolist()

        from transformers import BertTokenizer

        # トークナイザーの読み込み
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # テキストデータのトークン化
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        import torch
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

        # ラベルをテンソルに変換
        labels = torch.tensor(labels)

        # データセットの作成
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

        # データローダーの作成
        train_dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=8)


        from transformers import BertForSequenceClassification, AdamW

        # モデルの読み込み
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

        # オプティマイザーの準備
        optimizer = AdamW(model.parameters(), lr=2e-5)

        from torch.nn import CrossEntropyLoss

        # モデルをトレーニングモードに設定
        model.train()

        # エポック数の設定
        epochs = 1

        # トレーニングループ
        for epoch in range(epochs):
            for step, batch in enumerate(train_dataloader):
                b_input_ids, b_input_mask, b_labels = batch
                
                # 勾配をゼロに設定
                model.zero_grad()
                
                # 出力を取得
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits
                
                # ロスの逆伝播
                loss.backward()
                
                # パラメータの更新
                optimizer.step()
                
                if step % 10 == 0:
                    print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")
        
        print('train completed')

        # # モデルを評価モードに設定
        # model.eval()

        # # 予測の実行
        # with torch.no_grad():
        #     outputs = model(**inputs)
        #     logits = outputs.logits

        # # ロジットをラベルに変換
        # predictions = torch.argmax(logits, dim=1).tolist()

        # # 予測結果をDataFrameに追加
        # test_df['predictions'] = predictions
        # print(test_df)

