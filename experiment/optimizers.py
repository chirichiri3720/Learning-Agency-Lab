from model import BertModel,RobertaModel,DebertaModel,DistilbertModel,XLNetModel,ElectoraModel

def get_optimizer_grouped_parameters(model,model_name, lr=1e-3, weight_decay=0.01, lr_decay=0.95):
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    if model_name == 'deberta':
        model_part = model.model.deberta  
    elif model_name == 'deberta':
        model_part = model.model.deberta  
    elif model_name == 'bert':
        model_part = model.model.bert
    elif model_name == 'roberta':
        model_part = model.model.roberta
    elif model_name == 'distilbert':
        model_part = model.model.distilbert
    elif model_name == 'xlnet':
        model_part = model.model.transformer
    elif model_name == 'albert':
        model_part = model.model.albert
    elif model_name == 'electora':
        model_part = model.model.electora
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    layers = [model_part.embeddings] + list(model_part.encoder.layer)
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
