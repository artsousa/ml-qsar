import json
import logging
import os
import sys
import importlib
from typing import Dict, List, Tuple, Union, Iterable

import dgl
import numpy as np
import torch
import torch.nn as nn

from src.utils import (
    aux, 
    dataset,
    metrics as mt, 
)

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

from torch_geometric.utils import (
    unbatch,
    unbatch_edge_index
)

importlib.reload(mt)
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

def get_batched_samples(task: FSMolTask, inner_batch: int) -> Tuple:
    support_data = TFGraphBatchIterable(
                        samples=task.train_samples,
                        shuffle=True,              
                        max_num_nodes=10000,
                        max_num_graphs=inner_batch,
                    )
    evaluation_data = TFGraphBatchIterable(
                        samples=task.test_samples,
                        shuffle=True,              
                        max_num_nodes=10000,
                        max_num_graphs=inner_batch,
                    )

    return support_data, evaluation_data

def run_iteration(
        adaptation_data: TFGraphBatchIterable,
        evaluation_data: TFGraphBatchIterable,
        learner, 
        criterion: nn.Module,
        inner_steps: int,
        device: torch.device,
        metrics: List[str],
        no_grad: bool = False,
        n_edges: int = 3
    ) -> Tuple:

    step = 0
    flag = False
    train_loss = 0

    try:
        for _ in range(inner_steps):
            for inner_features, inner_labels in adaptation_data:            
                data = aux.prepare_data(inner_features, inner_labels, n_edges)

                for idx, batched in enumerate(data): # will run only once
                    x, y = batched
                    x, y = x.to(device), y.to(device)

                    out = learner(x)
                    loss = criterion(out, y.unsqueeze(1))
                    loss /= y.shape[0]
                    learner.adapt(loss, allow_unused=True)

                    train_loss += loss
                
                if (step + 1) % inner_steps == 0:
                    flag = True
                    break
                step += 1

            if flag:
                break

        valid_loss = 0
        output = np.array([]).reshape(0, 1)
        real_out = np.array([]).reshape(0, 1)
        for outer_features, outer_labels in evaluation_data:
            data = aux.prepare_data(outer_features, outer_labels, n_edges)

            for idx, batched in enumerate(data): # only once always
                x, y = batched
                x, y = x.to(device), y.to(device).unsqueeze(1)

                if no_grad:
                    with torch.no_grad():
                        out = learner(x)
                        loss = criterion(out, y)
                else:
                    out = learner(x)
                    loss = criterion(out, y)
                
                valid_loss += loss / y.shape[0]
                out = out.detach().numpy()
                output = np.concatenate((output, out), axis=0)
                real_out = np.concatenate((real_out, y), axis=0)

        valid_metrics = mt.calculate_metrics(
            metrics=metrics,
            y_true=real_out,
            y_pred=output,
            threshold=0,
        )

        torch.cuda.empty_cache()

        return valid_loss, valid_metrics, train_loss

    except Exception as e:
        logger.info(f"skipped, problem loading graph")
        return (0, 0, 0)

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
    inner_batch_size = 16
    steps_epoch = dataset.get_num_fold_tasks(DataFold.TRAIN) // task_batch_size
    logger.info(f"Total of {steps_epoch} steps per epoch\n")

    stub_graph_dataset = FSMolStubGraphDataset()
    description = stub_graph_dataset.get_batch_tf_data_description()

    ########## Meta training ##########
    meta_epoch_loss = 0.0
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
                logging.info(f" ==> Meta_training_loss: {meta_epoch_loss} \n")
                meta_epoch_loss = 0.0
        
        ### sample task_batch_size tasks ###
        task_batch = next(batches_iterator) 

        for i, task in enumerate(task_batch):
            learner = meta_learner.clone()
            inner_data, outer_data = get_batched_samples(task, inner_batch_size)

            # inner_data -> is the batched support data iterator
            # outer_data -> is the batched evaluation data iterator
            # edge_features_* -> not utilized here
            # node_features -> (N, z) N-nodes num in graphs batch, z-feature size
            # node_to_graph_map -> (N,) indicate each node belong to each graph
            # num_graphs_in_batch -> self explicative
            # adjacency_list_* -> list adjacency for all graphs in batch

            outer_loss, outer_metrics, inner_loss = run_iteration(
                                                        adaptation_data=inner_data,
                                                        evaluation_data=outer_data,
                                                        learner=learner,
                                                        criterion=criterion,
                                                        inner_steps=inner_steps,
                                                        device=device,
                                                        metrics=metrics,
                                                        n_edges=stub_graph_dataset.num_edge_types                      
                                                    )

            iteration_loss += outer_loss
            meta_train_loss += outer_loss.item() if type(outer_loss) is torch.Tensor else outer_loss 
            meta_training_metrics = {
                k: v + outer_metrics[k]
                for k, v in meta_training_metrics.items()
            }

            logger.info(f"task: {task.name}, [{i}] -> loss: {outer_loss}")
        
        optimizer.zero_grad()
        iteration_loss.backward()
        optimizer.step()

        meta_train_loss /= meta_batch_size
        summary_dict["meta_training_loss"].append(meta_train_loss)
        logger.info(f"meta_training_loss: {meta_train_loss}")

        meta_epoch_loss += meta_train_loss

        # Print some metrics
        for k, v in meta_training_metrics.items():
            v /= meta_batch_size
            summary_dict["meta_training_metrics"][k].append(v)
            logger.info(f"{k}: {v}")

        with open(os.path.join(save_path, "summary.json"), "w") as f:
            logger.info(summary_dict)
            json.dump(summary_dict, f)

        if (step + 1) % ckpt_steps == 0:
            save_checkpoint(
                model=meta_learner.module, save_path=save_path, step=step + 1
            )
            logger.info("Checkpoint saved.")

        torch.cuda.empty_cache()
        logger.info('--------------------------------------------')











