import torch
import numpy as np
import logging

from typing import Dict, Tuple
from torch import Tensor

logger = logging.getLogger()


def get_batch_index(adj_matrix: Tensor, 
                    graph_map: Tensor,
                    num_graphs: int) -> Tensor:

    batch_index = torch.ones((adj_matrix.shape[0],), dtype=torch.int32) * -1

    for j in range(num_graphs):
        ind = set(np.where(graph_map == j)[0])

        # for i, edge in enumerate(adj_matrix):
        #     edge = np.array(edge)
        #     logger.info(f"edge set: {edge}")
        #     logger.info(f"inters: {ind.intersection(edge)}")
        #     logger.info(f"edge set: {ind.intersection(edge) == set(edge)}")
        #     break

        ind_adj = [i for i, edge in enumerate(adj_matrix) 
                       if ind.intersection(np.array(edge)) == set(np.array(edge))]

        batch_index[ind_adj] = j
        

    return batch_index

def map_batch(features: Dict, num_edge: int) -> Tuple:

    adjacency_lists = tuple(torch.tensor(features[f"adjacency_list_{edge_type_idx}"], dtype=torch.long)
                                for edge_type_idx in range(num_edge)),

    batch_indexes = tuple()
    for adj_list in adjacency_lists[0]:
        
        batch_indexes = (*batch_indexes, 
                        get_batch_index(adj_list, 
                                        Tensor(features['node_to_graph_map']), 
                                        features['num_graphs_in_batch']))

    return batch_indexes, adjacency_lists[0]