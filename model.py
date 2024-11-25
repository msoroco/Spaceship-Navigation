import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_shape : list, n_actions : int, n_convs=2, kernel_size=5, pool_size=2, n_out_channels=16, n_lins=3):
        """
        Parameters
        ----------
        state_shape : list of int
            The shape of an input state as (channels, height, width)
        n_actions : int
            The number of actions
        n_convs : int, optional
            The number of Conv2d layers
        kernel_size : int, optional
            The kernel size of the Conv2d layers
        pool_size : int, optional
            The size of the pooling layers
        n_out_channels: int, optional
            The number of channels at the end of the convolutional layers
        n_lins: int, optional
            The number of linear layers
        """
        super(DQN, self).__init__()
        state_shape = np.array(state_shape)

        # conv layer setup
        channels = np.linspace(state_shape[0], n_out_channels, num=n_convs+1).astype(int)
        conv_layers = []
        for i in range(n_convs):
            conv_layers.append(nn.Conv2d(channels[i], channels[i+1], kernel_size))
            state_shape[1:] = state_shape[1:] - (kernel_size - 1)
            if i < n_convs-1: 
                conv_layers.append(nn.MaxPool2d(pool_size))
                state_shape[1:] = np.floor((state_shape[1:] - pool_size) / pool_size + 1)
            state_shape[0] = channels[i+1]
        self.convs = nn.Sequential(*conv_layers)

        # lin layer setup
        lin_sizes = np.linspace(np.prod(state_shape), n_actions, num=n_lins+1).astype(int)
        lin_layers = []
        for i in range(n_lins):
            lin_layers.append(nn.Linear(lin_sizes[i], lin_sizes[i+1]))
            if i < n_lins-1: 
                lin_layers.append(nn.ReLU())
        self.lins = nn.Sequential(*lin_layers)

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.lins(x)
        return x
    

class FullyConnectedDQN(nn.Module):
    def __init__(self, state_shape : list, n_actions : int, layers=[64, 64, 32, 16]):
        """
        Parameters
        ----------
        state_shape : list of int
            The shape of an input state as (channels, height, width)
        n_actions : int
            The number of actions
        layers : [int], optional
            The size of each hidden layers
        """
        super(FullyConnectedDQN, self).__init__()
        state_shape = np.array(state_shape)

        # lin layer setup
        input_size = np.prod(state_shape)
        lin_sizes = np.hstack(([input_size], layers, [n_actions]))
        lin_layers = []
        for i in range(len(layers)+1):
            lin_layers.append(nn.Linear(lin_sizes[i], lin_sizes[i+1]))
            if i < len(layers): 
                lin_layers.append(nn.ReLU())
        self.lins = nn.Sequential(*lin_layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.lins(x)
        return x
    


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Activation for the first hidden layer
        x = F.relu(self.fc2(x))  # Activation for the second hidden layer
        x = F.softmax(self.fc3(x), dim=-1)  # Output probabilities
        return x
