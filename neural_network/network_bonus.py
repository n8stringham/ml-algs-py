# This program uses pytorch to implement a feed forward NN

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, activation, initialization, num_layers):
        super(Model, self).__init__()

        # set hyperparams 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.activation = activation
        self.initialization = initialization

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # network architecture
        dim = self.input_dim
        for _ in range(self.num_layers):
            self.layers.append(nn.Linear(dim, self.hidden_dim))
            dim = self.hidden_dim
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            if self.initialization == 'He':
                nn.init.kaiming_uniform_(m.weight)
            elif self.initialization == 'Xavier':
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        # apply activation after each layer except for output
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        output = self.layers[-1](x)
        return output 
