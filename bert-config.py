from transformers import BertTokenizer, BertForSequenceClassification,BertConfig

# モデル名
model_name = 'bert-base-uncased'

# 設定ファイルのダウンロード
config = BertConfig.from_pretrained(model_name)
config.save_pretrained('./bert-base-uncased-config')