import torch
import numpy as np

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import List, Any, Tuple
from torch.utils.data import DataLoader, _utils

from src.utils import dataset

default_collate = _utils.collate.default_collate


def build_dataloader(
    x: List[np.array],
    batch_size: int,
    y: np.ndarray = None,
    shuffle: bool = False,
    num_workers: int = 2,
) -> torch.utils.data.DataLoader:
    """Create pytorch dataloader
    Parameters
    ----------
    x : List[np.ndarray]
        List of [adjacency, node features]
    batch_size : int
        Batch size
    y : np.ndarray, optional
        Labels for training, by default None
    shuffle : bool, optional
        Shuffle data, by default False
    num_workers : int, optional
        Number of workers for DataLoader, by default 2
    Returns
    -------
    torch.utils.data.DataLoader
        Instantiated DataLoader
    """
    dataset = dataset.GraphDataset(x=x, y=y)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=post_batch_padding_collate_fn,
    )

    return dataloader


def post_batch_padding_collate_fn(batch_data: List[tuple]) -> Tuple[torch.Tensor]:
    """Custom collate function for PyTorch DataLoader.
    This function is used to zero pad inputs to the same dimentions after batching.
    
    Arguments:
        batch_data {list} -- list of tuple (x, y) from GraphDataset
            - x: list of torch tensors [adj_mat, node_feat, atom_vec]
            - y: torch tensor
    Parameters
    ----------
    batch_data : List[tuple]
        List of tuples (x, y) with length of batch size.
        Each element in the list is an instance from GraphDataset.
            - x: List[torch.Tensor]
            - y: torch.Tensor
    Returns
    -------
    Tuple[torch.Tensor]
        Tuple (x, y) of padded and batched tensors.
    """
    x, y = zip(*batch_data)
    
    adj_mat, node_feat, atom_vec = zip(*x)
    num_atoms = [len(v) for v in atom_vec]
    padding_final = np.max(num_atoms)
    padding_len = padding_final - num_atoms
    adj_mat = torch.stack(
        [F.pad(a, (0, p, 0, p), "constant", 0) for a, p in zip(adj_mat, padding_len)], 0
    )
    node_feat = torch.stack(
        [F.pad(n, (0, 0, 0, p), "constant", 0) for n, p in zip(node_feat, padding_len)], 0
    )
    atom_vec = torch.stack(
        [F.pad(v, (0, 0, 0, p), "constant", 0) for v, p in zip(atom_vec, padding_len)], 0
    )
    x = [adj_mat, node_feat, atom_vec]
    y = torch.stack(y, 0)

    return x, y