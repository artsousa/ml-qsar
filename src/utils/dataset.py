"""PyTorch Dataset for graphs"""

from typing import Optional, Tuple, List

import dgl
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset

import logging
logger = logging.getLogger()


class MolecularGraphDataset(dgl.data.DGLDataset):
    def __init__(self,  x: List[List[torch.tensor]], y: Optional[np.array]):
        self.EDGE_TYPE = {0: "simple", 1: "double", 2: "triple"}
        self.ETYPES = [('_N', val, '_N') for key, val in self.EDGE_TYPE.items()]
        self.data = self._load_graphs(x, y)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def _load_graphs(self, x: List[List[np.array]], y: Optional[np.array]):
        graphs = []
        for adj_mats, feat_nodes in zip(x[0], x[1]):

            graph = dgl.heterograph({self.ETYPES[0]: (adj_mats[0][0, :], 
                                                adj_mats[0][1, :])}, )

            for idx, adj_matrix in enumerate(adj_mats):
                if idx == 0:
                    continue

                graph.add_edges(adj_matrix[0, :], adj_matrix[1, :], 
                                etype=self.EDGE_TYPE[idx])

            # graph.add_nodes(feat_nodes.shape[0])
            # graph.ndata['h'] = torch.tensor(feat_nodes, dtype=torch.long)
            
            
            graphs.append(graph)

            logger.info(f"graph: {graph.is_valid()}")

        return graphs

class GraphDataset(Dataset):
    def __init__(self, x: List[List[np.array]], y: Optional[np.array] = None):
        """PyTorch Dataset for molecular graphs.
        Parameters
        ----------
        x : List[List[np.array]]
            List of adjacency matrices and node feature matrices.
            Must be in the following order [adj_mat, node_feat] where
            each of adj_mat and node_feat is a list of np.ndarray.
        y : np.ndarray, optional
            Labels, by default None
        """
        super(GraphDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x[0])

    @property
    def num_tasks(self):
        if len(self.y.shape) == 1:
            return 1
        else:
            return self.y.shape[-1]

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Get item from dataset by indexing.
        Parameters
        ----------
        idx : int
            Item index in dataset
        Returns
        -------
        Tuple[List[torch.Tensor], torch.Tensor]
            Tuple of 2 elements (x, y).
            x is a list of [adj_mat, node_feat, atom_vec] with shapes:
                - adj_mat: (N, N)
                - node_feat: (N, F)
                - atom_vec: (N, 1)
            where N is number of nodes and F is number of features.
            y is the output labels. y is set to 0 if no outputs are provided.
        """

        x = [self.x[0][idx], torch.FloatTensor(self.x[1][idx])]
        a = torch.ones(len(x[1]), 1)
        x.append(a)

        try:
            y = torch.tensor(self.y[idx])
        except TypeError:
            y = torch.tensor(0)
        finally:
            if len(y.size()) < 1:
                y = y.unsqueeze(-1)

        return x, y