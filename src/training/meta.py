import json
import logging
import os
import sys
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src import utils

from fs_mol.data import (
    FSMolDataset
)

def meta_training(
    meta_learner,
    meta_steps: int,
    meta_batch_size: int,
    dataset: FSMolDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    inner_steps: int,
    device: torch.device,
    save_path: str,
    ckpt_steps: int,
    metrics: List[str] = ["accuracy_score"],
) -> None:
    
    pass
    # if meta_batch_size > len(loaders["meta_train"]["train"]):
    #     raise ValueError(
    #         f"meta_batch_size ({meta_batch_size}) must <= number of tasks ({len(loaders['meta_train']['train'])})"
    #     )