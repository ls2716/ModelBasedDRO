"""Implement a neural network policy for the synthetic voting problem."""

import torch
import torch.nn as nn

class NeuralNetworkPolicy(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the neural network policy.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output actions.
        """
        super(NeuralNetworkPolicy, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.softmax(x)
