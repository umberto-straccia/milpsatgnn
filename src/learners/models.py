import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torchinfo import summary
import networkx as nx

import torch
from torch_geometric.data import Data, Batch
import random

import learners.common as LC
import learners.modules as MOD

class GCNPolicy(nn.Module):
    def __init__(self, 
                 emb_size, 
                 n_cons_f, 
                 n_edge_f, 
                 n_var_f, 
                 n_layers=2, 
                 activation_name="relu", 
                 aggregation="mean",
                 bias=True, 
                 is_graph_level=True, 
                 dev=False, 
                 name=None, 
                 graph_op=pyg.nn.global_mean_pool,
                 bip_layer_anticipate_mp=False,
                 bip_conv_order="cons-var",  # this one makes sense only when bip_layer_anticipate_mp=True
                 bip_conv_append_activation=False,
                 force_bias_value=False,
                 ):
        super(GCNPolicy, self).__init__()
        self.emb_size = emb_size
        self.n_cons_f = n_cons_f
        self.n_edge_f = n_edge_f
        self.n_var_f = n_var_f
        self.activation_name = activation_name
        self.aggregation = aggregation
        self.bias = bias
        self.n_layers = n_layers
        self.is_graph_level = is_graph_level
        self.dev = dev
        if self.dev:
            print("----> DEV MODE SET <----")
        self.name = name
        self.graph_level_op = graph_op
        self.bip_layer_anticipate_mp = bip_layer_anticipate_mp
        self.bip_conv_order = bip_conv_order
        self.conv_append_activation = bip_conv_append_activation
        self.force_bias_value = force_bias_value
        self._build_layers()
    #<

    def print_info(self):
        print(20 * "-")
        print("- GCNPolicy:")
        print("\t - n_cons_f:", self.n_cons_f)
        print("\t - n_edge_f:", self.n_edge_f)
        print("\t - n_var_f:", self.n_var_f)
        print("\t - emb_size:", self.emb_size)
        print("\t - n_layers:", self.n_layers)
        print("\t - activation:", self.activation_name)
        print("\t - aggregation:", self.aggregation)
        print("\t - anticipate_mp:", self.bip_layer_anticipate_mp)
        print("\t - conv_order:", self.bip_conv_order)
        print(20 * "-")
        print("** SUMMARY **")
        summary(self)
        summary(self.bp_layers[0])


    def _build_layers(self):
        
        # Ok, line 118
        # TODO: initialization with orthogonal
        self.cons_embedding = nn.Sequential(
            nn.Linear(self.n_cons_f, self.emb_size),
            LC.activations[self.activation_name]
        )

        # OK, line 123
        self.edge_embedding = nn.Identity()

        # OK, line 127
        # TODO: initialization with orthogonal
        self.var_embedding = nn.Sequential(
            nn.Linear(self.n_var_f, self.emb_size),
            LC.activations[self.activation_name]
        )

        # OK, BipartiteLayer computes:
        #    - v_to_c
        #    - c_to_v
        self.bp_layers = nn.ModuleList([
            MOD.BipartiteLayer(self.emb_size, activation_name=self.activation_name, 
                           aggregation=self.aggregation, bias=self.bias, dev=self.dev,
                           anticipate_mp=self.bip_layer_anticipate_mp,
                           conv_order=self.bip_conv_order,
                           conv_append_activation=self.conv_append_activation,)
            for _ in range(self.n_layers)
        ])

        if self.is_graph_level:
            # Ok, line 140: activation after first then linear
            self.output_module = nn.Sequential(
                nn.Linear(2 * self.emb_size, self.emb_size),
                LC.activations[self.activation_name],
                nn.Linear(self.emb_size, 1)
            )
        else:
            assert False, f"Only graph level is supported for now"
        
        self._reset_parameters()
    #<

    def _reset_parameters(self):
        for layer in self.bp_layers:
            layer._reset_parameters()
        #<
        # init all layers with a diagonal matrix
        # for layer in [self.cons_embedding[0], self.var_embedding[0], self.output_module[0], self.output_module[2]]:
        #     layer.weight.data = torch.eye(layer.in_features, layer.out_features)

        # calculate gain
        gain = nn.init.calculate_gain(self.activation_name)
        # initialize using gain
        torch.nn.init.orthogonal_(self.cons_embedding[0].weight, gain=gain)
        torch.nn.init.orthogonal_(self.var_embedding[0].weight, gain=gain)
        torch.nn.init.orthogonal_(self.output_module[0].weight, gain=gain)
        torch.nn.init.orthogonal_(self.output_module[2].weight, gain=gain)
        
        # in all layers, set bias to 0 if present
        for layer in [self.cons_embedding[0], self.var_embedding[0], 
                      self.output_module[0], self.output_module[2]]:
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
        
        # nn.init.xavier_uniform_(self.cons_embedding[0].weight)
        # nn.init.xavier_uniform_(self.var_embedding[0].weight)
        # nn.init.xavier_uniform_(self.output_module[0].weight)
        # nn.init.xavier_uniform_(self.output_module[2].weight)
    #<

    def forward(self, batch): # , cons_x, var_x, x_edge, edge_index): 
        # adj matrix E_{i, j}: constraint i to variable j
        # constrints: V
        # variables: W
            
        cons_features = self.cons_embedding(batch["left"].x)
        # n_cons x emb_size
        
        var_features = self.var_embedding(batch["right"].x)  # q_0(h_j ^ W)
        # n_vars x emb_size
        
        edge_features = self.edge_embedding(batch["left", "to", "right"].edge_attr)
        # n_dedges x 1 (identity)
        
        for i in range(self.n_layers):
            cons_features, var_features = \
                self.bp_layers[i](cons_features, var_features, batch)
            
        
        if self.is_graph_level:
            # graph level pooling using pytorch geometric
            cons_features = self.graph_level_op(cons_features, batch["left"].batch)
            var_features = self.graph_level_op(var_features, batch = batch["right"].batch)
            out = torch.cat([cons_features, var_features], dim=1)
        else:
            assert False, "not yet developed"

        out = self.output_module(out)
        return out
    #<
#< GCNPolicy

