"""
Gated Graph Neural Network module for graph classification tasks
"""
import dgl
import torch
import logging
from torch import nn

from dgl.nn.pytorch import GatedGraphConv, GlobalAttentionPooling

logger = logging.getLogger(__name__)

class GraphClsGGNN(nn.Module):

    def __init__(self, in_feats, out_feats, n_steps, 
                    n_etypes, n_layers=2, num_cls=1, p_dropout=0.2):
        super(GraphClsGGNN, self).__init__()
        
        self.num_cls = num_cls
        self.convs = nn.ModuleList([])
        for i in range(n_layers):
            if i == 0:
                self.convs.append(GatedGraphConv(
                    in_feats=in_feats,
                    out_feats=out_feats,
                    n_steps=n_steps,
                    n_etypes=n_etypes))
            else:
                self.convs.append(GatedGraphConv(
                    in_feats=out_feats,
                    out_feats=out_feats,
                    n_steps=n_steps,
                    n_etypes=n_etypes))

        # self.pooling = GlobalAttentionPooling(nn.Linear(out_feats, 1))
        self.output_layer = nn.Linear(out_feats, num_cls)
        self.dropout = nn.Dropout(p=p_dropout)
        self.activation = nn.ReLU()
        
        self.reset_parameters()

    def reset_parameters(self):
        for l in self.convs:
            l.reset_parameters()

        for k, v in self.state_dict().items():
            if "output_layer" in k:
                if "weight" in k:
                    nn.init.xavier_uniform_(v)
                elif "bias" in k:
                    nn.init.zeros_(v)

    def forward(self, graph):
        etypes = graph.edata['_TYPE']
        feats = graph.ndata["h"].float()

        for layer in self.convs:
            out = layer(graph, feats, etypes=etypes)
            self.activation(out)

        with graph.local_scope():
            graph.ndata['h'] = out
            hg = dgl.mean_nodes(graph, 'h')
            logits = self.output_layer(hg)
            # preds = torch.argmax(logits, -1)

            return logits









