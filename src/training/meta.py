import json
import logging
import os
import sys
import importlib
from typing import Dict, List, Tuple, Union, Iterable

import numpy as np
import torch
import torch.nn as nn

from src.utils import aux
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
from fs_mol.data.maml import (
    FSMolStubGraphDataset, 
    TFGraphBatchIterable
)
from fs_mol.data.fsmol_task_sampler import SamplingException

importlib.reload(aux)
logger = logging.getLogger()


def save_checkpoint(model: nn.Module, save_path: str, step: Union[str, int]):
    ckpt_name = f"step_{step}"
    torch.save(model, os.path.join(save_path, f"{ckpt_name}.pth"))
    torch.save(
        model.state_dict(),
        os.path.join(save_path, f"{ckpt_name}_state_dict.pth"),
    )

def get_task_sampler(train_size, 
                        min_test_size,
                        test_size) -> nn.Module:

    return StratifiedTaskSampler(
                train_size_or_ratio=train_size,
                valid_size_or_ratio=0,
                test_size_or_ratio=(min_test_size, test_size),
            )

def run_iteration(
        adaptation_data: Tuple[torch.Tensor],
        evaluation_data: Tuple[torch.Tensor],
        learner,
        criterion: nn.Module,
        inner_steps: int,
        device: torch.device,
        metrics: List[str],
        no_grad: bool = False,
    ) -> Tuple:

    pass


def get_batched_samples(task: FSMolTask) -> Tuple:
    support_data = TFGraphBatchIterable(
                        samples=task.train_samples,
                        shuffle=True,              
                        max_num_nodes=10000,
                        max_num_graphs=4,
                    )
    evaluation_data = TFGraphBatchIterable(
                        samples=task.test_samples,
                        shuffle=True,              
                        max_num_nodes=10000,
                        max_num_graphs=4,
                    )

    return support_data, evaluation_data

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

    task_sampler = get_task_sampler(train_size, min_test_size, test_size)

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

    stub_graph_dataset = FSMolStubGraphDataset()
    description = stub_graph_dataset.get_batch_tf_data_description()

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
        
        ### sample task_batch_size tasks ###
        task_batch = next(batches_iterator) 

        for task in task_batch:
            logging.info(f'{task.name} {len(task.train_samples)}')

            learner = meta_learner.clone()
            inner_data, outer_data = get_batched_samples(task)

            # inner_data -> is the batched support data iterator
            # outer_data -> is the batched evaluation data iterator
            # edge_features_* -> not utilized here
            # node_features -> (N, z) N-nodes num in graphs batch, z-feature size
            # node_to_graph_map -> (N,) indicate each node belong to each graph
            # num_graphs_in_batch -> self explicative
            # adjacency_list_* -> list adjacency for all graphs in batch

            for inner_features, inner_labels in inner_data:
                # logging.info(f"keys: {inner_features.keys()}")

                node_features = torch.tensor(inner_features['node_features'], 
                                            dtype=torch.int32)
                node2graphmap = torch.tensor(inner_features['node_to_graph_map'],
                                            dtype=torch.int32)
                
                batch_index, adj_lists = aux.map_batch(inner_features,
                                                        stub_graph_dataset.num_edge_types)

                logger.info(f"match: {len(adj_lists)}, {len(batch_index)}")
                # logging.info(f"node_features: {node_features.shape}")
                # logging.info(f"node to graph: {node_to_graph_map.shape}")
                # logging.info(f"adjacency_list_ 0: {adjacency_list_0.shape}")
                # logging.info(f"adjacency_list_ 1: {adjacency_list_1.shape}")
                # logging.info(f"adjacency_list_ 2: {adjacency_list_2.shape}")

                break

            break
            # outer_loss, outer_metrics = run_iteration(
            #                                 adaptation_data=inner_data,
            #                                 evaluation_data=outer_data,
            #                                 learner=learner,
            #                                 criterion=criterion,
            #                                 inner_steps=inner_steps,
            #                                 device=device,
            #                                 metrics=metrics,                         
            #                             )

        break

        
        logging.info(f" ==> Meta_training_loss: {meta_train_loss} \n")













