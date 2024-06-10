from model import Bertmodel

def get_optimizer_grouped_parameters(model, lr=1e-3, weight_decay=0.01, lr_decay=0.95):
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
   
    bert_model = model.model.bert
    num_layers = bert_model.config.num_hidden_layers
    layers = [bert_model.embeddings] + list(bert_model.encoder.layer)
    layers.reverse()
    lr_layer = lr#層ごとの学習率の初期値

    for layer in layers:
        lr_layer *= lr_decay 
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr_layer,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr_layer,
            },
        ]
    
    optimizer_grouped_parameters += [
        {
            "params": [p for n, p in model.named_parameters() if 'embeddings' not in n and 'encoder' not in n],
            "weight_decay": weight_decay,
            "lr": lr,
        },
    ]

    
    return optimizer_grouped_parameters
