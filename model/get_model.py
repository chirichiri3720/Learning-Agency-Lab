from experiment.utils import set_seed

from .model import Bertmodel, Debertamodel
from .gbm import XGBoostClassifier, LightGBMClassifier, CBTClassifier

def get_classifier(name, *, device, model_config, num_labels=6, seed=42):
    set_seed(seed=seed)
    if name == "bert":
        return Bertmodel(device=device, num_labels=num_labels, model_config=model_config)
    if name == "deberta":
        return Debertamodel(device=device, num_labels=num_labels, model_config=model_config)
    else:
        raise KeyError(f"{name} is not defined.")

def get_tree_classifier(name, *, input_dim, output_dim, model_config, seed=42, verbose=0):
    if name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "lightgbm":
        return LightGBMClassifier(input_dim, output_dim, model_config, verbose)
    elif name == "catboost":
        return CBTClassifier(input_dim, output_dim, model_config, verbose)
    else:
        raise KeyError(f"{name} is not defined")