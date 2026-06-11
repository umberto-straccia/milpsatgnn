import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
import networkx as nx

import torch
from torch_geometric.data import Data, Batch
import random
from torchinfo import summary

import learners.common as LC

class BipartiteGraphConvolution(MessagePassing):
    def __init__(self, 
                 emb_size, 
                 activation_name="relu", 
                 aggregation='mean', 
                 right_to_left=False, 
                 bias=True,
                 conv_append_activation=False,
                 dev=False, name=None, 
                 force_bias_value=False):
        super(BipartiteGraphConvolution, self).__init__(aggr=aggregation, flow="source_to_target")
        
        self.emb_size = emb_size
        self.activation_name = activation_name
        self.aggregation = aggregation
        self.right_to_left = right_to_left
        self.bias = bias
        self.conv_append_activation = conv_append_activation
        self.dev = dev
        self.name = name
        self.force_bias_value = force_bias_value
        self._build_layers()
    #<


    def _build_layers(self):
#        print("---> WARN: creating feature_module_left with bias=True !!! THE ACTUAL PARAMETER IS IGNORED")
        self.feature_module_left = nn.Sequential(
            # bias set to true as in https://github.com/liujl11git/GNN-MILP/blob/main/models.py
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            LC.activations[self.activation_name],
            # below bias is True, as in Keras when not speciefied
            nn.Linear(self.emb_size, self.emb_size)
        )
                
        self.feature_module_edge = nn.Identity()

#        print("---> WARN: creating feature_module_right with bias=False !!! THE ACTUAL PARAMETER IS IGNORED")
        self.feature_module_right = nn.Sequential(
            # bias set to false as in https://github.com/liujl11git/GNN-MILP/blob/main/models.py
            nn.Linear(self.emb_size, self.emb_size, bias=self.force_bias_value),
            LC.activations[self.activation_name],
            # below bias is True, as in Keras when not speciefied
            nn.Linear(self.emb_size, self.emb_size)  # here bias is True, as in Keras when not spec
        )

        if self.conv_append_activation:      
            self.feature_module_left.append(LC.activations[self.activation_name])
            self.feature_module_right.append(LC.activations[self.activation_name])

        # Output module
        self.output_module = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            LC.activations[self.activation_name],
            nn.Linear(self.emb_size, self.emb_size)
        )

        self._reset_parameters()
    #< _build_layers


    def _reset_parameters(self):
        for module in [self.feature_module_left, self.feature_module_right, self.output_module]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # nn.init.xavier_uniform_(layer.weight)
                    # calculate gain
                    gain = nn.init.calculate_gain(self.activation_name)
                    torch.nn.init.orthogonal_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
    #<
    
    def forward(self, x_left, x_right, batch):
        left_features = self.feature_module_left(x_left)
        right_features = self.feature_module_right(x_right)
        
        edge_index = batch["left", "to", "right"].edge_index
        # XXX: check here: embedding of edge_attr?
        edge_attr = batch["left", "to", "right"].edge_attr if hasattr(batch["left", "to", "right"], "edge_attr") else None
        
        if self.right_to_left:
            #source, target = edge_index[1], edge_index[0]
            edge_index = edge_index[[1, 0]]
            src_features = right_features
            tgt_features = left_features
        else: # left to right
            src_features = left_features
            tgt_features = right_features
        #<

        out = self.propagate(x=(src_features, tgt_features),
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            size=(src_features.size(0), tgt_features.size(0)), 
                            )
        
        out = self.output_module(torch.cat([out, tgt_features], dim=1))

        return out
    #<

    
    # message to node i:
    #   (j, i) in E and flow=source_to_target
    #   (i, j) in E and flow=target_to_source
    def message(self, x_j, edge_attr=None):
        # if self.dev:
        #     print(f"--{self.name}-- in messsage(x_j, ...): x_j = {x_j}, shape = {x_j.shape}")
        # if edge_attr is not None:
        #     return self.feature_module_edge(edge_attr) * x_j
        # else:
        #     return x_j
        
        # print("Message:")
        # print("x_j.shape", x_j.shape)
        # print("edge_attr.shape", edge_attr.shape)
        return edge_attr.view(-1, 1) * x_j
    #<

    # def update(self, aggr_out):
    #     print("---> update: unimplemented")
    #     pass

#<


# ------------------------------------------------------

class BipartiteLayer(nn.Module):
    def __init__(self, emb_size, activation_name="relu", aggregation="sum", bias=True, anticipate_mp=False, conv_order="cons-vars", 
                 conv_append_activation=False, dev=False, force_bias_value=False):
        super(BipartiteLayer, self).__init__()
        print("---> BipartiteLayer: actual parameter 'bias' is ignored")
        self.emb_size = emb_size
        
        self.activation_name = activation_name
        
        self.aggregation = aggregation
        self.bias = bias
        self.anticipate_mp = anticipate_mp
        self.conv_order = conv_order
        self.conv_append_activation = conv_append_activation
        self.dev = dev
        self.force_bias_value = force_bias_value
        self._build_layers()
        self.activation = LC.activations[self.activation_name]
        
    #<


    def _build_layers(self):
        # (self, emb_size, activation_name="relu", aggregation='mean', right_to_left=False, bias=True, dev=False)
        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, 
                                                     activation_name=self.activation_name, 
                                                     aggregation=self.aggregation, 
                                                     right_to_left=True, 
                                                     bias=self.bias,
                                                     conv_append_activation=self.conv_append_activation,
                                                     dev=self.dev, name="var-to-cons", 
                                                     force_bias_value=self.force_bias_value) 

        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, 
                                                     activation_name=self.activation_name, 
                                                     aggregation=self.aggregation, 
                                                     right_to_left=False, 
                                                     bias=self.bias,
                                                     conv_append_activation=self.conv_append_activation,
                                                     dev=self.dev, name="cons-to-var", 
                                                     force_bias_value=self.force_bias_value)
        self._reset_parameters()
    #<

    def _reset_parameters(self):
        self.conv_v_to_c._reset_parameters()
        self.conv_c_to_v._reset_parameters()
    #<



    def forward_with_anticipation(self, prev_cons_features, prev_var_features, batch):
        if self.conv_order == "cons-vars":
            constraint_features = self.conv_v_to_c(prev_cons_features, prev_var_features, batch)
            constraint_features = self.activation(constraint_features)
            
            variable_features = self.conv_c_to_v(constraint_features, prev_var_features, batch)
            variable_features = self.activation(variable_features)
        elif self.conv_order == "vars-cons":
            variable_features = self.conv_c_to_v(prev_cons_features, prev_var_features, batch)
            variable_features = self.activation(variable_features)
            constraint_features = self.conv_v_to_c(prev_cons_features, variable_features, batch)
            constraint_features = self.activation(constraint_features)
        else:
            raise ValueError(f"Unknown conv_order: {self.conv_order}. Expected 'cons-vars' or 'vars-cons'.")
        return  constraint_features, variable_features
    #<


    # formula 35 in overleaf
    def forward(self, prev_cons_features, prev_var_features, batch):
        if self.anticipate_mp:
            constraint_features, variable_features = self.forward_with_anticipation(prev_cons_features, prev_var_features, batch)
        else:
            constraint_features = self.conv_v_to_c(prev_cons_features, prev_var_features, batch)
            constraint_features = self.activation(constraint_features)

            variable_features = self.conv_c_to_v(prev_cons_features, prev_var_features, batch)
            variable_features = self.activation(variable_features)
        #<     
        
        return constraint_features, variable_features
    #<