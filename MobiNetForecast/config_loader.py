from pathlib import Path
from itertools import product
from typing import Dict, Any, Tuple
import re
import yaml

default_config = {
    "validation_ratio": 0.1,
    "delimiter": " ",
    "min_input_length": 10,
    "max_input_length": 14,
    "test_prediction_length": 5,
    "batch_size": 128,
    "device": "cpu",
    "max_epochs": 10,
    "block_size": 24,
    "learning_rate": 5.e-3,
    "weight_decay": 5.e-1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "decay_lr": True,
    "warmup_iters": 200,
    "lr_decay_epochs": 10,
    "min_lr": 5.e-7,
    "seed": 42,
    "data_dir": "./data",
    "dataset": "geolife",
    "n_layer": 12,
    "n_head": 6,
    "n_embd": 512,
    "bias": False,
    "dropout": 0.1,
    "model_checkpoint_directory": "./models/",
    "train_from_checkpoint_if_exist": False,
    "custom_initialization": False,
    "patience": 3,
    "continuity": True,
    "beam_width": 5,
    "store_predictions": False,
    "resolution": 7,
    "window_size": 300,
}

def load_config(config_file: str | Path) -> dict:
    """
    Load a YAML configuration file and apply default values for missing parameters.

    Parameters:
    - config_file (str | Path): Path to the configuration YAML file.

    Returns:
    - config_list (dict): A dictionary with the final configuration, including defaults.
    """
    with open(config_file, 'r', encoding='utf-8') as stream:
        try:
            config_list = yaml.safe_load(stream)

            if config_list is None:
                config_list = {}

            for config_name, config_values in config_list.items():
                if config_values is None:
                    config_values = {}

                for key, value in default_config.items():
                    config_values[key] = config_values.get(key, value)

                config_list[config_name] = config_values

            return config_list

        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None

def save_config(name: str, config: dict, path: Path) -> None:
    """
    Save a configuration dictionary to a YAML file.

    Parameters:
    - name (str): The name of the configuration.
    - config (dict): The configuration dictionary to save.
    - path (Path): The file path where the configuration should be saved.
    """
    try:
        with path.open('w', encoding='utf-8') as file:
            yaml.dump({name: config}, file, default_flow_style=False, allow_unicode=True)
    except (OSError, yaml.YAMLError) as e:
        print(f"Error saving configuration: {e}")


def expand_config(config_list: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    For each base config in `config_list`, look at the keys
    'n_embd','n_head','n_layer','learning_rate','min_lr'.
    If a value is a list, treat it as multiple choices; otherwise
    wrap the scalar as a single-element list.  Then do a full
    Cartesian product over those five axes, yielding one config
    per combination, and give each a unique name suffix.
    Returns a flat dict: new_name -> new_config_dict.
    """
    hyperparams = ['n_embd', 'n_head', 'n_layer', 'learning_rate', 'min_lr', 'weight_decay']
    expanded: Dict[str, Dict[str, Any]] = {}

    for base_name, base_cfg in config_list.items():
        choices = []
        for hp in hyperparams:
            val = base_cfg.get(hp, None)
            if isinstance(val, (list, tuple)):
                choices.append(list(val))
            else:
                choices.append([val])

        for combo in product(*choices):
            combo_dict = dict(zip(hyperparams, combo))
            parts = [base_name] + [
                f"emb={combo_dict['n_embd']}",
                f"head={combo_dict['n_head']}",
                f"layer={combo_dict['n_layer']}",
                f"lr={combo_dict['learning_rate']:.0e}",
                f"minlr={combo_dict['min_lr']:.0e}",
                f"wd={combo_dict['weight_decay']:.0e}",
            ]
            new_name = "_".join(parts)
            expanded[new_name] = { **base_cfg, **combo_dict }

    return expanded


_prefix_map = {
    'emb':  'n_embd',
    'head': 'n_head',
    'layer':'n_layer',
    'lr':   'learning_rate',
    'minlr':'min_lr',
    'wd':   'weight_decay',
}

def parse_config_name(name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Reverse the naming scheme:
      - Walk tokens split on '_', from right → left.
      - As long as token matches "<short>=<number>" for one of our prefixes,
        strip it off and parse it into params.
      - Stop as soon as a token doesn’t match: everything to the left is base_name.
    """
    parts = name.split('_')
    params: Dict[str, Any] = {}
    i = len(parts) - 1

    while i >= 0:
        if len(params) >= len(_prefix_map):
            break

        token = parts[i]
        if '=' not in token:
            break

        short, val_str = token.split('=', 1)
        if short not in _prefix_map:
            break

        if re.search(r'[eE\.]', val_str):
            val: Any = float(val_str)
        else:
            val = int(val_str)

        params[_prefix_map[short]] = val
        i -= 1

    base_name = "_".join(parts[:i+1])
    params = {'name': base_name, **params}
    return params
