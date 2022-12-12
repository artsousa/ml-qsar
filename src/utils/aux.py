import sys
import torch
import numpy as np
import logging
import importlib
import torch.nn.functional as F

from src.utils import dataset
from fs_mol.data.maml import (
    FSMolStubGraphDataset,
)

from typing import Dict, Tuple, List
from torch import Tensor
from torch.utils.data import (
    Dataset,
    DataLoader
)
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

    unbatched_lists = [unbatch_edge_index(torch.tensor(adj_list.T, dtype=torch.int64), 
                                            torch.tensor(node2graphmap, dtype=torch.int64)) 
                            for adj_list in adj_lists]

    # 3 loops iteration, 1 for each adJ-list, num_graphs for each adj
    adj_per_sample = {}
    for unb_list in unbatched_lists:
        sample_counter = 0

        while sample_counter < inner_features['num_graphs_in_batch']:
            try:
                if len(adj_per_sample[sample_counter]) > 0:
                    adj_per_sample[sample_counter].append(unb_list[sample_counter])
            except KeyError as e:
                adj_per_sample[sample_counter] = [unb_list[sample_counter]]
            except IndexError as ie:
                adj_per_sample[sample_counter].append(torch.tensor([]))
            except Exception as e:
                logger.info(f'Exception while unbatching stuff: {e}')
            finally: 
                sample_counter += 1

    list_adj = []
    for key in adj_per_sample.keys():
        list_adj.append([np.array(to_dense_adj(adj_list)[0]) 
                                if adj_list.shape[0] > 0 else np.array([])
                                for adj_list in adj_per_sample[key]])

    # if adj_list.shape[0] > 0 else np.array([]) 
    # logger.info(f"samples: {len(ds[0][0][0])}") 
    ds = dataset.GraphDataset([list_adj, nodes_unbatched], 
                                inner_labels['target_value'])

    dataloader = DataLoader(
        dataset=ds,
        batch_size=inner_features['num_graphs_in_batch'],
        num_workers=0,
        shuffle=False,
        collate_fn=post_batch_padding_collate_fn,
    )

    for batched_data in dataloader:
        logger.info(f"BATCHED DATA: {batched_data}")




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
    
    logger.info(f'')

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
    












