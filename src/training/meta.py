import json
import logging
import os
import sys
from typing import Dict, List, Tuple, Union, Iterable

import numpy as np
import torch
import torch.nn as nn

from src import utils
from dpu_utils.utils import RichPath
from more_itertools import (
    chunked, 
    sample
) 

from fs_mol.data import (
    DataFold,
    FSMolDataset,
    FSMolTask,
    FSMolTaskSample,
    StratifiedTaskSampler,
)
from fs_mol.data.fsmol_task_sampler import SamplingException


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
    task_batch_size: int,
    train_size: int,
    min_test_size: int, 
    test_size: int,
    metrics: List[str] = ["accuracy_score"],
) -> None:
    
    total_train_tasks = dataset.get_num_fold_tasks(DataFold.TRAIN)
    if meta_batch_size > total_train_tasks:
        raise ValueError(
            f"meta_batch_size ({meta_batch_size}) must <= number of tasks ({total_train_tasks})"
        )

    task_sampler = StratifiedTaskSampler(
        train_size_or_ratio=train_size,
        valid_size_or_ratio=0,
        test_size_or_ratio=(min_test_size, test_size),
    )

    def read_and_sample_from_task(paths: List[RichPath], 
                                    id: int) -> Iterable[FSMolTaskSample]:
        for i, path in enumerate(paths):
            task = FSMolTask.load_from_file(path)
            try:
                yield task_sampler.sample(task, seed=id + i)
            except SamplingException as e:
                logger.debug(f"Sampling task failed:\n{str(e)}")

    summary_dict = {}
    summary_dict["meta_training_loss"] = []
    summary_dict["meta_training_metrics"] = {m: [] for m in metrics}

    save_checkpoint(model=meta_learner.module, save_path=save_path, step="init")
    logger.info("Initial checkpoint saved.")

    epoch_count = 0
    steps_epoch = dataset.get_num_fold_tasks(DataFold.TRAIN) // task_batch_size
    logger.info(f"Total of {steps_epoch} steps per epoch\n")

    ########## Meta training ##########
    for step in range(meta_steps):

        logging.info(f"(Epoch {epoch_count + 1}) Step {step}")

        iteration_loss = 0.0
        meta_train_loss = 0.0
        meta_training_metrics = {m: 0 for m in metrics}

        ### reset train tasks iterable          ###
        ### reset train tasks batches iterable  ### 
        if (step + 1) % steps_epoch == 0 or step == 0:
            train_task_samples = dataset.get_task_reading_iterable(
                                        data_fold=DataFold.TRAIN,
                                        task_reader_fn=read_and_sample_from_task,                        
                                    )
            batches_iterator = chunked(train_task_samples, n=task_batch_size)

            if step > 0:
                epoch_count += 1
        
        ### sample K tasks ###
        task_batch = next(batches_iterator)

        for task in task_batch:
            
                
        # for task_batch in :
        #     logging.info(f"task batch size: {len(task_batch)}, {type(task_batch)}")    
        #     print(task_batch)

        
        logging.info(f" ==> Meta_training_loss: {meta_train_loss} \n")













