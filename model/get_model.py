from experiment.utils import set_seed

from .model import Bertmodel, Debertamodel

def get_classifier(name, *, device, model_config, num_labels=6, seed=42):
    set_seed(seed=seed)
    if name == "bert":
        return Bertmodel(device=device, num_labels=num_labels, model_config=model_config)
    if name == "deberta":
        return Debertamodel(device=device, num_labels=num_labels, model_config=model_config)
    else:
        raise KeyError(f"{name} is not defined.")