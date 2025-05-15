import os
import glob
import time
import pickle
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from MobiNetForecast.TrajectoryBatchDataset import TrajectoryBatchDataset
from MobiNetForecast.CausalLM import ModelConfig, CausalLM
from MobiNetForecast.evaluator import evaluate_model, evaluate_collision_prediction
from MobiNetForecast.Trainer import Trainer
from MobiNetForecast.logger import get_logger
from MobiNetForecast.config_loader import save_config, parse_config_name
from baselines import TrajectoryAttnLSTM, TrajectoryLSTM, TrajectoryGRU, TrajectoryMarkovChain, TrajectoryEvolveGCN


def setup_environment(seed: int) -> None:
    """
    Set up the environment by configuring CUDA and setting random seeds.

    Args:
    - seed (int): The seed for random number generators.
    - device_id (str): The CUDA device ID to set for training.
    """
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_dataset(
    config: Dict[str, Any],
    test_mode: bool = False,
    graph_mode: bool = False
) -> Union[TrajectoryBatchDataset, pd.DataFrame]:
    """
    Load the trajectory dataset based on configuration.

    Args:
    - config (Dict[str, Any]): Configuration dictionary.
    - test_mode (bool): Whether to load test or training data (default is False).

    Returns:
    - TrajectoryBatchDataset: The dataset object.
    """
    dataset_path = Path(config["data_dir"]) / config["dataset"]\
        / f"resolution-{config['resolution']}"
    if graph_mode:
        df_path = dataset_path / ("edges_train.csv" if test_mode else "edges_test.csv")
        dataset = pd.read_csv(df_path)
    else:
        dataset = TrajectoryBatchDataset(
            dataset_path,
            dataset_type='test' if test_mode else 'train',
            delimiter=config["delimiter"],
            validation_ratio=config["validation_ratio"]
        )
        config["vocab_size"] = dataset.vocab_size
    return dataset


def load_model(
    model: Union[torch.nn.Module, TrajectoryMarkovChain],
    checkpoint_path: Optional[Path],
    device: str
) -> torch.nn.Module:
    """
    Load a model from a checkpoint.

    Args:
    - model (torch.nn.Module): The model to be loaded.
    - checkpoint_path (Path): Path to the checkpoint file.
    - device (str): The device to load the model onto (e.g., 'cpu' or 'cuda').

    Returns:
    - Module: The initialized model, possibly with loaded weights.
    """
    if isinstance(model, TrajectoryMarkovChain):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        optimizer = None
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        optimizer = checkpoint['optimizer']
    config = checkpoint['config']
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, _ in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    return model, config, optimizer

def initialize_model(
    config: Dict,
    custom_init: Optional[torch.Tensor] = None,
) -> CausalLM:
    """
    Initializes a Causal Language Model (CausalLM) using the provided configuration.

    Args:
        config (dict): A dictionary containing model configuration parameters:
        custom_init (Optional[torch.Tensor]): A PyTorch tensor for custom model initialization.

    Returns:
        CausalLM: An instance of the Causal Language Model initialized with the given parameters.
    """
    model_config = ModelConfig(
        block_size=config["block_size"],
        vocab_size=config["vocab_size"],
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        dropout=config["dropout"],
        bias=config["bias"]
    )
    return CausalLM(model_config, custom_init)


def train_model(
    name: str,
    dataset: Union[TrajectoryBatchDataset, pd.DataFrame],
    config: Dict[str, Any],
    model: Optional[Union[torch.nn.Module, TrajectoryMarkovChain]] = None
) -> None:
    """
    Set up and execute the training process.

    Args:
    - name (str): Name for the current training session (used for saving logs/checkpoints).
    - dataset (Union[TrajectoryBatchDataset, pd.DataFrame]): Dataset object for training.
    - config (Dict[str, Any]): Configuration dictionary.
    - model (Optional[torch.nn.Module]): The model to be trained (can be None before loading).
    """
    time_str = time.strftime("%Y%m%d-%H%M%S")
    model_checkpoint_directory = Path(config["model_checkpoint_directory"]) / (name + "-" + time_str)
    log_directory = model_checkpoint_directory / 'logs'
    log_directory.mkdir(parents=True, exist_ok=True)

    save_config(name, config, log_directory / 'config.yaml')

    if model is None:
        if config['custom_initialization']:
            custom_init_path = os.path.join(config["data_dir"], config["dataset"], f"resolution-{config['resolution']}", 'embeddings.npy')
            embeddings_np = np.load(custom_init_path)
            custom_init = torch.from_numpy(embeddings_np).to(torch.float32)
            model = initialize_model(config, custom_init=custom_init)
        else:
            model = initialize_model(config=config)

    if config['train_from_checkpoint_if_exist']:
        glob_pattern = str(Path(config["model_checkpoint_directory"]) / (name + "-" + "[0-9]"*8 + "-" + "[0-9]"*6))
        model_checkpoints = sorted(glob.glob(glob_pattern))
        if len(model_checkpoints) > 0:
            last_checkpoint = Path(model_checkpoints[-1]) / 'checkpoint.pt'
            model, config, optimizer = load_model(model, last_checkpoint, config['device'])

    logger = get_logger(log_directory, name, phase="train")
    logger.info("Training instance initiated at %s", time_str)

    if isinstance(model, (TrajectoryMarkovChain, TrajectoryEvolveGCN)):
        model.train(dataset, logger, str(model_checkpoint_directory))
    else:
        model.to(config["device"])
        trainer = Trainer(model, dataset, config, logger, str(model_checkpoint_directory))
        trainer.train()

def select_model(
    baseline: Optional[str],
    config: Dict[str, Any]
) -> Optional[Union[torch.nn.Module, TrajectoryMarkovChain]]:
    """
    Select the appropriate model based on the specified baseline.
    Args:
    - baseline (Optional[str]): The baseline model to be used.
    - config (Dict[str, Any]): Configuration dictionary.
    Returns:
    - Optional[Union[torch.nn.Module, TrajectoryMarkovChain]]: The selected model instance.
    """
    if baseline is None:
        return None

    if baseline == "gru":
        return TrajectoryGRU(config)
    elif baseline == "lstm":
        return TrajectoryLSTM(config)
    elif baseline == "lstm-attn":
        return TrajectoryAttnLSTM(config)
    elif baseline == "mc":
        return TrajectoryMarkovChain(config)
    elif baseline == "evolve-gcn":
        return TrajectoryEvolveGCN(config)
    else:
        raise ValueError(f"Unknown baseline: {baseline!r}")

def save_search(search_dict: Dict[str, Any]) -> None:
    """
    Save the search results to a CSV file.
    Args:
    - search_dict (Dict[str, Any]): Dictionary containing the search results.
    """
    search_dict = [{**parse_config_name(name), **result} for name, result in search_dict.items()]
    out_path = "search_results.csv"
    pd.DataFrame(search_dict)\
       .to_csv(out_path, mode='a', header=not os.path.isfile(out_path), index=False)

def test_model(
    name: str,
    dataset: Union[TrajectoryBatchDataset, pd.DataFrame],
    config: Dict[str, Any],
    model: Optional[torch.nn.Module] = None
) -> list:
    """
    Set up and execute the testing process.

    Args:
    - name (str): Name of the configuration (used for loading the model checkpoint).
    - dataset (Union[TrajectoryBatchDataset, pd.DataFrame]): Dataset object for testing.
    - config (Dict[str, Any]): Configuration dictionary.
    - model (Optional[torch.nn.Module]): The model to be tested (can be None before loading).
    """
    glob_pattern = str(Path(config["model_checkpoint_directory"]) / (name + "-" + "[0-9]"*8 + "-" + "[0-9]"*6))
    saved_checkpoints = sorted(glob.glob(glob_pattern))
    if len(saved_checkpoints) == 0:
        model_checkpoint_directory = config['model_checkpoint_directory']
        print("No saved model found in the checkpoint directory:", model_checkpoint_directory)
        return
    elif len(saved_checkpoints) > 1:
        timestamp = saved_checkpoints[-1].split(f"{name}-")[-1]
        time_string = datetime.strptime(timestamp, "%Y%m%d-%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        print(f"Multiple saved models found. Using the latest one (saved at {time_string}).")
    model_checkpoint_directory = saved_checkpoints[-1]

    log_directory = Path(model_checkpoint_directory) / 'logs'
    logger = get_logger(log_directory, name, phase="test")
    logger.info("Evaluation instance initiated at %s", time.strftime("%Y%m%d-%H%M%S"))
    config["vocab_size"] = dataset.vocab_size
    if model is None:
        model = initialize_model(config)

    checkpoint_path = Path(model_checkpoint_directory) / 'checkpoint.pt'
    model, _, __ = load_model(model, checkpoint_path, config['device'])

    dataset.create_test_batches(config["batch_size"], config["test_prediction_length"])

    if isinstance(model, TrajectoryMarkovChain):
        model.evaluate(dataset, logger)
        return evaluate_collision_prediction(config, logger)
    elif isinstance(model, (TrajectoryEvolveGCN, )):
        model.evaluate(dataset, logger)
        return evaluate_collision_prediction(config, logger, generate_edges=False)
    else:
        model.to(config["device"])
        trajectory_prediction_result = evaluate_model(model, dataset, config, logger)
        collision_prediction_result = evaluate_collision_prediction(config, logger)
        return {**trajectory_prediction_result, **collision_prediction_result}
