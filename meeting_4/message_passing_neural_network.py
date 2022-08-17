import numpy as np

import torch
from torch.nn import Embedding, Linear
from torch_scatter import scatter

from qm9_dataset import QM9Dataset


class MessagePassingNeuralNetwork(torch.nn.Module):
    """ pytorch message passing neural network for transferable modeling of PESs"""

    def make_update_layer(self):

        # input : atomic message vector
        input_size = self.message_dim

        # output : atomic hidden state vector
        output_size = self.embedding_dim

        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=input_size, affine=True),
            torch.nn.Linear(in_features=input_size, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=output_size),
        )

    def make_readout_layer(self):

        # input : concatenation of hidden state vectors at each message pass iteration
        input_size = self.embedding_dim * (self.npass + 1)

        # output : atomic energy
        output_size = 1

        return torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=self.embedding_dim * (self.npass + 1), affine=True),
            torch.nn.Linear(in_features=self.embedding_dim * (self.npass + 1), out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=200),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=200, out_features=1),
        )


    def __init__(self,
                 atoms=[1, 6, 7, 8, 9],
                 distance_cutoff=3.0,
                 embedding_dim=8,
                 npass=0):
        super(MessagePassingNeuralNetwork, self).__init__()

        self.atoms = atoms
        self.shifts = torch.from_numpy(np.arange(0.8, distance_cutoff+1e-5, 0.1).astype(np.float32))
        self.embedding_dim = embedding_dim
        self.npass = npass

        nshifts = len(self.shifts)

        # IMPORTANT : this dimension is dependent on your message function implementation
        # when you change the message function, you have to change this variable
        # otherwise, you'll get an error
        self.message_dim = nshifts + self.embedding_dim

        # embeds a scalar nuclear charge into a vector of length embedding dim
        self.embedding = Embedding(max(self.atoms) + 1, self.embedding_dim)

        # the MPNN update function (one per iteration)
        self.update_layers = [self.make_update_layer() for _ in range(self.npass)]
        self.update_layers = torch.nn.ModuleList(self.update_layers)

        # the MPNN message function doesn't use a neural network
        # consider changing this
        # self.message_layers = 

        # the MPNN readout function 
        self.readout_layer = self.make_readout_layer()
        self.readout_layer[-1].bias.data.fill_(0.0)


    def forward(self, batch):
        """ Definition of a forward pass for this network """

        ##########
        # inputs #
        ##########
    
        # z_i : [natom, 1]
        z_i = batch.x

        # edge_index : [2, nedge]
        edge_index = batch.edge_index
        e_source = edge_index[0]
        e_sink = edge_index[1]

        # dist_ij : [nedge, 1]
        dist_ij = batch.edge_attr

        # mol_ind : [natom]
        mol_ind = batch.batch

        natom = z_i.shape[0]
        nedge = edge_index.shape[1]

        ###############################
        # form radial basis functions #
        ###############################

        # radial basis functions (~symmetry functions) are a way of encoding scalar distances

        # rbf_ij : [nedge, nshift]
        rbf_ij = dist_ij - self.shifts.reshape((1, -1))
        rbf_ij = torch.exp(-1.0 * torch.square(rbf_ij))

        ########################
        # atom type embeddings #
        ########################

        # embed the nuclear charge (Z) in a vector

        # x_i_0 : [natom, nembed]
        x_i_0 = self.embedding(z_i)
        x_i_0 = x_i_0.reshape((-1, self.embedding_dim))

        ################
        # message pass #
        ################

        # save the hidden state vectors of each message passing iteration
        x_i_list = [x_i_0]

        for t in range(self.npass):

            # the last iteration's hidden state vector
            x_i_prev = x_i_list[-1]

            #~~ change this message function! ~~#
            # message from j -> i is a concatenation of x_j and dist(i, j)
            m_ij = torch.cat([rbf_ij, x_i_prev[e_sink]], dim=1)

            # sum over j to get aggregated messages for each i
            m_i = scatter(m_ij, e_source.reshape((-1,1)), dim=0)

            # update the hidden state
            # lots of options here too
            x_i_next = x_i_prev + 0.1 * self.update_layers[t](m_i)

            # keep track of the hidden states
            x_i_list.append(x_i_next)

        ###########
        # readout #
        ###########

        # concatenate the hidden states over all of the iterations
        x_i_all = torch.cat(x_i_list, dim=1)

        # readout atomic energies via a dense feed-forward NN
        y_i = self.readout_layer(x_i_all)

        # sum atom energies into molecule energies
        y = scatter(y_i, mol_ind.reshape(-1,1), dim=0)

        return y
