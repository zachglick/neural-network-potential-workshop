import torch

class PESNeuralNetwork(torch.nn.Module):
    """ pytorch dense feed-forward neural network for modeling a Potential Energy Surface"""

    def __init__(self, input_size):
        super(PESNeuralNetwork, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=50),
            torch.nn.Linear(in_features=50, out_features=20),
            torch.nn.Linear(in_features=20, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=1),
        )

    def forward(self, x):
        """definition of a forward pass for this network"""
        return self.linear_relu_stack(x)
