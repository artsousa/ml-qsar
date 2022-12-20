#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import math
from typing import Any, List, Set, Dict, Tuple, Optional, Iterable, Union
import warnings

import torch
from src.layers import gatedgraphconv as ggc

warnings.simplefilter("ignore")


class GGNeuralNet(torch.nn.Module):

    def __init__(
        self,
        num_features, 
        num_layers,
        aggr = 'add',
        bias = True,
        num_edge_types=1,
    ) -> None:
        super(GGNeuralNet, self).__init__()
        
        self.graph_conv = ggc.GatedGraphConv(
                                num_features, 
                                num_layers, 
                                aggr=aggr, 
                                bias=bias,
                                num_edge_types=num_edge_types,)

    def get_layers_dim(self) -> None:
        return self.graph_conv


        


