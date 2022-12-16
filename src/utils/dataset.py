"""PyTorch Dataset for graphs"""

from typing import Optional, Tuple, List

import dgl
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
import importlib

importlib.reload(dgl)

import logging
logger = logging.getLogger()


class MolecularGraphDataset(dgl.data.DGLDataset):
    def __init__(self,  x: List[List[torch.tensor]], y: Optional[np.array]):
        self.EDGE_TYPE = {0: "simple", 1: "double", 2: "triple"}
        self.ntype = '_N'
        self.ETYPES = [(self.ntype, val, self.ntype) 
                        for key, val in self.EDGE_TYPE.items()]
        
        self.y = y
        self.data = self._load_graphs(x)
        
    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]

    def __len__(self):
        return len(self.data)

    def _load_graphs(self, x: List[List[np.array]]):
        c = 0
        graphs = []
        for adj_mats, feat_nodes in zip(x[0], x[1]):
            num_nodes = -1
            graph_data = {}  
            
            for idx, adj_matrix in enumerate(adj_mats):
                try:
                    graph_data[self.ETYPES[idx]] = (adj_matrix[0, :], 
                                                    adj_matrix[1, :])
                except IndexError as ie:
                    graph_data[self.ETYPES[idx]] = ([], [])
            try: 
                graph = dgl.heterograph(graph_data,)
                graph.nodes[self.ntype].data['h'] = torch.tensor(feat_nodes, dtype=torch.long)
                graphs.append(graph)
                c += 1

            except dgl.DGLError as dglerror:
                if self.y.shape[0] > 0:
                    self.y = np.delete(self.y, c)
                    c -= 1
                else:
                    self.y = np.array([])
                    graphs = []

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