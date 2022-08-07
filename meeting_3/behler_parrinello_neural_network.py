import torch

from qm9_dataset import QM9Dataset

import numpy as np

class BehlerParrinelloNeuralNetwork(torch.nn.Module):
    """ pytorch dense feed-forward neural network for transferable modeling of PESs"""

    def atom_network(self, input_size):
        """ Create a dense feed-forward neural network for modeling atomic energies
            for a single atom type

            Args: 
                input_size (int) : the length of the atomic feature vector

            Returns:
                atom_network (torch.nn.Sequential) : the atomic energy network

        """

        network = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=500),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=500, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=100),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=100, out_features=1),
        )

        # b/c targets normalized w/ linear regression, pre-set the bias of each network to 0.0
        network[-1].bias.data.fill_(0.0)

        return network

    def __init__(self, input_size, atoms):
        super(BehlerParrinelloNeuralNetwork, self).__init__()

        self.atoms = atoms
        self.atom_networks = torch.nn.ModuleList([self.atom_network(input_size) for atom in atoms])

    def forward(self, x, ind, e):
        """ Definition of a forward pass for this network """

        for atom_i, atom in enumerate(self.atoms):
            
            atom_model = self.atom_networks[atom_i]
            atom_energies = atom_model(x[atom])

            e.index_add_(0, ind[atom], atom_energies)

        return e
