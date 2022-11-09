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
    DataFold,
    FSMolDataset,
)

logger = logging.getLogger()


def save_checkpoint(model: nn.Module, save_path: str, step: Union[str, int]):
    ckpt_name = f"step_{step}"
    torch.save(model, os.path.join(save_path, f"{ckpt_name}.pth"))
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f"{ckpt_name}_state_dict.pth"),
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
    
    total_train_tasks = dataset.get_num_fold_tasks(DataFold.TRAIN)
    if meta_batch_size > total_train_tasks:
        raise ValueError(
            f"meta_batch_size ({meta_batch_size}) must <= number of tasks ({total_train_tasks})"
        )

    summary_dict = {}
    summary_dict["meta_training_loss"] = []
    summary_dict["meta_training_metrics"] = {m: [] for m in metrics}

    save_checkpoint(model=meta_learner.module, save_path=save_path, step="init")
    logger.info("Initial checkpoint saved.")

    for step in range(meta_steps):
        logger.info(f"Iteration {step}")

        iteration_loss = 0.0
        meta_train_loss = 0.0
        meta_training_metrics = {m: 0 for m in metrics}

        # Meta training

        logging.info(f"meta_training_loss: {meta_train_loss}")
