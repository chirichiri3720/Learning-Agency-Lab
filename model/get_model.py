from experiment.utils import set_seed

from .model import BertModel, DebertaModel,RobertaModel,DistilbertModel,XLNetModel,ALbertModel,ElectoraModel

def get_classifier(name, *, device, model_config, num_labels=6, seed=42):
    set_seed(seed=seed)
    if name == "bert":
        return BertModel(device=device, num_labels=num_labels, model_config=model_config)
    if name == "deberta":
        return DebertaModel(device=device, num_labels=num_labels, model_config=model_config)
    if name == "roberta":
        return RobertaModel(device=device, num_labels=num_labels, model_config=model_config)
    if name == "distilbert":
        return DistilbertModel(device=device, num_labels=num_labels, model_config=model_config)
    if name == "xlnet":
        return XLNetModel(device=device, num_labels=num_labels, model_config=model_config)
    if name == "albert":
        return ALbertModel(device=device, num_labels=num_labels, model_config=model_config)
    if name == 'electora':
        return ElectoraModel(device=device, num_labels=num_labels, model_config=model_config)
    else:
        raise KeyError(f"{name} is not defined.")