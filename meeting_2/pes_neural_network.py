import torch

class PESNeuralNetwork(torch.nn.Module):
    """ pytorch dense feed-forward neural network for modeling a Potential Energy Surface"""

    def __init__(self, input_size):
        super(PESNeuralNetwork, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

    def forward(self, x):
        """definition of a forward pass for this network"""
        return self.linear_relu_stack(x)
