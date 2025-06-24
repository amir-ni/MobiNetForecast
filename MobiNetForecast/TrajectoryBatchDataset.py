import os
import json
import random
from collections import defaultdict
from typing import List, Tuple, Union, Dict
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import IterableDataset


class TrajectoryBatchDataset(IterableDataset):
    """
    A dataset class for handling variable-length trajectory data, used in training,
    validation, and testing of sequence models with PyTorch.

    Args:
        dataset_directory (str): Path to the dataset directory.
        dataset_type (str): One of 'train', 'val', or 'test' indicating the dataset split.
        delimiter (str): Delimiter used in the trajectory sequences (default is space).
        validation_ratio (float): Ratio of the training data used for validation.
    """

    def __init__(
        self,
        dataset_directory: Union[str, Path],
        dataset_type: str = 'train',
        delimiter: str = ' ',
        validation_ratio: float = 0.1
    ):
        self.dataset_directory = dataset_directory

        if dataset_type in ['train', 'val']:
            self.dataX = []
            self.dataY = []
            train_df = pd.read_csv(os.path.join(dataset_directory, 'data_train.csv'))["HexagonSequence"]
            train_data = [np.array([int(j) for j in i.strip().split(delimiter)]) for i in train_df]
            number_of_trajectories = len(train_data)
            number_of_train_trajectories = int(number_of_trajectories * (1 - validation_ratio))
            if dataset_type == 'train':
                self.data = train_data[:number_of_train_trajectories]
            elif dataset_type == 'val':
                self.data = train_data[number_of_train_trajectories:]
        elif dataset_type == 'test':
            self.test_df = pd.read_csv(os.path.join(dataset_directory, 'data_test.csv'))
            self.dataX = [np.array([int(j) for j in i.strip().split(delimiter)]) for i in self.test_df["InputSequence"]]
            self.dataY = [np.array([int(j) for j in i.strip().split(delimiter)]) for i in self.test_df["PredictionSequence"]]
        else:
            raise ValueError('Invalid type')

        self.vocab_size = sum(1 for _ in open(
            os.path.join(dataset_directory, 'vocab.txt'), encoding='utf-8'))

        self.batches = []
        self.dataset_type = dataset_type

    def create_test_batches(self, batch_size: int, test_prediction_length: int) -> None:
        """
        Organizes test data into batches of similar sequence lengths.

        Args:
            batch_size (int): Number of samples per batch.
            test_prediction_length (int): Fixed length to which prediction sequences are padded.
        """
        size_to_indices = defaultdict(list)
        for i, x in enumerate(self.dataX):
            size_to_indices[len(x)].append(i)
        for size_indices in size_to_indices.values():
            for i in range(0, len(size_indices), batch_size):
                batch = size_indices[i:i+batch_size]
                self.batches.append(batch)
        self.dataY = [np.pad(a, (0, max(0, test_prediction_length - len(a))), mode='constant') for a in self.dataY]

    def create_batches(
        self,
        batch_size: int,
        observe: Union[int, List[int]],
        predict: Union[int, List[int]] = 1,
        shuffle: bool = True,
        drop_last: bool = False
    ) -> None:
        """
        Prepares training/validation batches from trajectories.

        Args:
            batch_size (int): Number of samples per batch.
            observe (Union[int, List[int]]): Length(s) of observation windows.
            predict (Union[int, List[int]]): Length(s) of prediction windows.
            shuffle (bool): Whether to shuffle batches.
            drop_last (bool): Whether to drop the last batch if it's smaller than batch_size.
        """
        if isinstance(observe, int):
            observe = [observe]
        if isinstance(predict, int):
            predict = [predict] * len(observe)

        for trajectory in self.data:
            for j, observe_length in enumerate(observe):
                for i in range(0, len(trajectory) - observe_length - predict[j] + 1):
                    self.dataX.append(trajectory[i:i+observe_length])
                    self.dataY.append(
                        trajectory[i+observe_length:i+observe_length+predict[j]])

        size_to_indices = defaultdict(list)
        for i, x in enumerate(self.dataX):
            size_to_indices[len(x)].append(i)

        batches = []
        for size_indices in size_to_indices.values():
            for i in range(0, len(size_indices), batch_size):
                batch = size_indices[i:i+batch_size]
                if len(batch) == batch_size or not drop_last:
                    batches.append(batch)

        if shuffle:
            random.shuffle(batches)

        self.batches = batches

    def get_neighbors(self) -> Dict[int, List[int]]:
        """
        Loads neighbor information for each node from `neighbors.json`.

        Returns:
            Dict[int, List[int]]: A dictionary mapping each node ID to a list of its neighbors.
        """
        with open(os.path.join(self.dataset_directory, 'neighbors.json'), encoding='utf-8') as neighbors_file:
            neighbors = json.load(neighbors_file)
            neighbors = {int(k): v + [0] for k, v in neighbors.items()}
            neighbors[0] = []
        return neighbors

    def get_mapping(self) -> Dict[int, str]:
        """
        Loads the mapping from node indices to original values.

        Returns:
            Dict[int, str]: A dictionary mapping index to original value.
        """
        with open(os.path.join(self.dataset_directory, 'mapping.json'), encoding='utf-8') as mapping_file:
            mapping = json.load(mapping_file)
            mapping = {int(v): k for k, v in mapping.items()}
        return mapping

    def __len__(self) -> int:
        """
        Returns the number of batches.

        Returns:
            int: Number of batches.
        """
        return len(self.batches)

    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Retrieves a batch by index.

        Args:
            index (int): Batch index.

        Returns:
            Tuple[torch.LongTensor, torch.LongTensor]: A tuple of input and target tensors.
        """
        batch_indices = self.batches[index]
        return (
            torch.LongTensor(np.stack([self.dataX[i] for i in batch_indices])),
            torch.LongTensor(np.stack([self.dataY[i] for i in batch_indices]))
        )

    def __iter__(self):
        """
        Yields batches of padded tensors during iteration.

        Yields:
            Tuple[torch.LongTensor, torch.LongTensor]: A tuple of input and target tensors for each batch.
        """
        for batch_indices in self.batches:
            max_length = max(len(self.dataY[i]) for i in batch_indices)
            padded_samples = [
                np.pad(self.dataY[i], (0, max_length - len(self.dataY[i])), mode='constant', constant_values=0)
                for i in batch_indices
            ]
            yield (
                torch.LongTensor(np.stack([self.dataX[i] for i in batch_indices])),
                torch.LongTensor(np.stack(padded_samples))
            )
