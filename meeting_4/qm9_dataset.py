import time
import numpy as np

from scipy.spatial import distance_matrix

import torch

import torch_geometric
from torch_geometric.data import Data, Dataset

from sklearn.linear_model import LinearRegression

class QM9Dataset(Dataset):
    """pytorch geometric Dataset class for handling QM9"""

    def __init__(self,
                 npz_file: str,
                 size: int=133885,
                 elems: [int]=[1, 6, 7, 8, 9],
                 distance_cutoff: float=5.0):
        """
        Args:
            npz_file (string): Path to the npz_file.
            size (int): Number of molecules to load
                For QM9, this can be up to 133,885
            elems ([int]): list of elements present in the dataset
                Elements are represented by their nuclear charge
            distance_cutoff (float) : maximum edge distance in molecular graph
                A larger distance_cutoff makes a denser molecular graph
        """

        # list of elements present in the dataset
        self.elems = elems
        self.distance_cutoff = distance_cutoff

        # the size of the whole QM9 dataset
        MAX_SIZE = 133885

        # shuffle the data (deterministic order)
        inds = np.arange(MAX_SIZE)
        np.random.seed(4201)
        np.random.shuffle(inds)

        # the fraction of the dataset to load here
        assert size <= MAX_SIZE
        inds = inds[-size:]

        # load the raw data from saved numpy file
        data = np.load(npz_file, allow_pickle=True)

        # Z (nuclear charges, Integer)      : (n_mol, n_atom)
        # R (raw molecular coordinate, Ang) : (n_mol, n_atom, 3)
        # E (energies, kcal / mol)          : (n_mol, 1)

        Z = data['nuclear_charges'][inds]
        R = data['coords'][inds]
        E = data['energies'][inds].reshape(-1,1)
        E = self.linear_regression(Z, E) * 627.509

        self.data_list = []
        for i in range(size):
            self.data_list.append(self.make_data(Z[i], R[i], E[i]))


    def make_data(self, Z, R, E):

        natom = len(Z)

        D = distance_matrix(R, R)
        edge_mask = np.where(np.logical_and(D < self.distance_cutoff, D > 0.0))
        edge_dist = D[edge_mask]
        edge_index = np.array([edge_mask[0], edge_mask[1]])

        x = torch.from_numpy(Z.reshape(natom, 1).astype(np.int32))
        edge_index = torch.from_numpy(edge_index.astype(np.int64))
        edge_attr = torch.from_numpy(edge_dist.reshape(-1, 1).astype(np.float32))
        y = torch.from_numpy(E.reshape(1, 1).astype(np.float32))
        pos = torch.from_numpy(R.astype(np.float32))

        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=y,
                    pos=pos)

        return data
        

    def linear_regression(self, Z, E):
        """ perform linear regression on molecular data

            Args:
                Z ([np.ndarray], int) nuclear charges : A list of molecules (specified by nuclear
                    charges). Each element of the list is an array of the nuclear charges in a 
                    molecule.
                E (np.ndarray, float) molecular energies : An array of molecular energies (Hartree).
                    This array is the same length as Z

            Returns:
                E_lr (np.ndarray, float) normalized molecular energies : The molecular energies (E)
                    with atomic energies (fit by linear regression) subtracted out.
        """

        Z_count = np.zeros((len(Z), len(self.elems)), dtype=np.int64)

        for ind_mol, Z_mol in enumerate(Z):
            for ind_elem, elem in enumerate(self.elems):
                Z_count[ind_mol, ind_elem] = np.sum(Z_mol == elem)

        lr = LinearRegression(fit_intercept=False)
        lr.fit(Z_count, E)
        E_lr = lr.predict(Z_count)

        std_pre_lr = np.std(E)
        std_post_lr = np.std(E - E_lr)

        print("\nNormalizing molecular energies with linear regression:")
        print(f"stddev(E) before linreg: {std_pre_lr:10.3f} Hartree / ({std_pre_lr*627.509:10.3f} kcal / mol)")
        print(f"stddev(E) after linreg:  {std_post_lr:10.3f} Hartree / ({std_post_lr*627.509:10.3f} kcal / mol)")

        self.lr_weights = lr.coef_

        return E - E_lr

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":

    # load (and generate features for) 100 random QM9 molecules 
    dataset = QM9Dataset("../meeting_3/QM9.npz", size=500)
