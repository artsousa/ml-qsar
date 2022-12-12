import sys
import torch
import numpy as np
import logging
import importlib

from src.utils import dataset
from fs_mol.data.maml import (
    FSMolStubGraphDataset,
)

from typing import Dict, Tuple, List
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.utils import (
    to_dense_adj,
    unbatch_edge_index
)

importlib.reload(dataset)
logger = logging.getLogger()


def get_batch_index(adj_matrix: np.array, 
                    graph_map: np.array,
                    num_graphs: int) -> np.array:

    batch_index = np.ones((adj_matrix.shape[0],), dtype=np.int) * -1

    for j in range(num_graphs):
        ind = set(np.where(graph_map == j)[0])

        ind_adj = [i for i, edge in enumerate(adj_matrix) 
                       if ind.intersection(np.array(edge)) == set(np.array(edge))]

        batch_index[ind_adj] = j

    return batch_index

def map_batch(features: Dict, num_edge: int, tomap: bool = False):

    adjacency_lists = tuple(np.array(features[f"adjacency_list_{edge_type_idx}"], dtype=np.int)
                                for edge_type_idx in range(num_edge)),

    if not tomap:
        return adjacency_lists[0]

    batch_indexes = tuple()
    for adj_list in adjacency_lists[0]:
        
        batch_indexes = (*batch_indexes, 
                        get_batch_index(adj_list, 
                                        features['node_to_graph_map'], 
                                        features['num_graphs_in_batch']))

    return batch_indexes, adjacency_lists[0]

def map_node(node_features: np.array, nodemap: np.array):

    assert node_features.shape[0] == nodemap.shape[0], \
        "Dimensions should match"

    return [node_features[np.where(nodemap == g_ind)] 
                    for g_ind in np.unique(nodemap)]


def prepare_data(inner_features: Dict, inner_labels: Dict, num_edges: int):

    node_features = inner_features['node_features']
    node2graphmap = inner_features['node_to_graph_map']
    nodes_unbatched = map_node(node_features, node2graphmap)

    adj_lists = map_batch(inner_features, num_edges)
    adj_lists = list(adj_lists)

    counter = 0
    unbatched_lists = [unbatch_edge_index(torch.tensor(adj_list.T, dtype=torch.int64), 
                                            torch.tensor(node2graphmap, dtype=torch.int64)) 
                            for adj_list in adj_lists]

    for idx, unb_list in enumerate(unbatched_lists):
        logger.info(f"{idx} LIST, ")

        for j, adj in enumerate(unb_list):
            logger.info(f"{j} node {adj} \n")

    logger.info(f"\n\n--\n\n")
    # 3 loops iteration, 1 for each adJ-list, num_graphs for each adj
    #
    adj_per_sample = {}
    for unb_list in unbatched_lists:
        logger.info(f'list: {len(unb_list)}')
        sample_counter = 0

        while sample_counter < inner_features['num_graphs_in_batch']:
            try:
                if len(adj_per_sample[sample_counter]) > 0:
                    adj_per_sample[sample_counter].append(unb_list[sample_counter])
            except KeyError as e:
                adj_per_sample[sample_counter] = [unb_list[sample_counter]]
            except Exception as e:
                logger.info(f'Exception while unbatching stuff: {e}')
            finally: 
                sample_counter += 1
        
    logger.info(f"size: {len(adj_per_sample)} \n\n")
    logger.info(f"st adj: {len(adj_per_sample[0])}")
    logger.info(f"st adj: {adj_per_sample[0]}\n\n")

    logger.info(f"nd adj: {len(adj_per_sample[1])}\n\n")
    logger.info(f"nd adj: {adj_per_sample[1]}\n\n")

    logger.info(f"th adj: {len(adj_per_sample[2])}\n\n")
    logger.info(f"th adj: {adj_per_sample[2]}\n\n")
    
    logger.info(f"nd adj: {len(adj_per_sample[3])}")
    logger.info(f"nd adj: {adj_per_sample[3]}")

    # for nodes_unb, adjlist_unb in zip(nodes_unbatched, unbatched_lists):
    #     logger.info(f"node_features: {nodes_unb.shape}, \n\n {len(adjlist_unb)}")
    # for adj in adj_lists:
    #     logger.info(f"adj * - {adj.shape}")
    #     logger.info(f"labels - {labels}")
    # logger.info(f"node features: {node_features.shape}")
    # dense one hot format to represent the graph
    # dense index format can be used to 

    # adj_matrices = []
    # for adj in adj_lists:
    #     if adj.shape[0] > 0:
    #         matrix = to_dense_adj(torch.tensor(adj, dtype=torch.int64))
    #         adj_matrices.append(matrix[0])
    #     else: 
    #         adj_matrices.append(torch.tensor([]))

    # for matrix in adj_matrices:
    #     logger.info(f"matrix shape 0: {matrix.shape}")

    # ds = dataset.GraphDataset([adj_matrices, node_features], labels)

    # logger.info(f"inner labels: {labels}")
    # logger.info(f"num_ tasks: {len(ds)}")

    # sample = ds[1]

    # logger.info(f"{len(sample)}")
    # for graph in sample[0]:
    #     logger.info(f"matrix shape: {graph.shape}")

    # logger.info(f"{np.sum(sample[0][0] - adj_matrices[0])}")

    # logger.info(f"first sample: {len(sample[0])} label: {sample[1]}")
    












