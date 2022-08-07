import time
import numpy as np
import scipy.spatial
import torch

from sklearn.linear_model import LinearRegression

SHIFTS = np.linspace(1.0, 5.0, 65)
ELEMS = [1, 6, 7, 8, 9]
WIDTH = 1.0

class QM9Dataset(torch.utils.data.Dataset):
    """pytorch Dataset class for handling QM9"""

    def __init__(self,
                 npz_file: str,
                 size: int=133885,
                 elems: [int]=[1, 6, 7, 8, 9],
                 shifts: [float]=np.linspace(1.0, 4.0, 33),
                 width: float=1.0):
        """
        Args:
            npz_file (string): Path to the npz_file.
            size (int): Number of molecules to load (up to 133,885)
        """

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

        self.n_feat = len(SHIFTS) * len(ELEMS)

        # pre-compute all features
        # this is potentially expensize
        t1 = time.time()
        X = [QM9Dataset.make_symmetry_functions(z, r) for z, r in zip(Z, R)]
        print(f"\nFeature generation complete: {time.time() - t1:.1f} seconds.")

        # normalize the features
        X = QM9Dataset.normalize_features(X, Z)

        # normalize the labels
        # save the scale so we can convert predictions back to kcal / mol
        self.scale = 1.0
        y = E * self.scale

        # convert to pytorch tensors
        self.y = y.astype(np.float32)
        self.X = [x.astype(np.float32) for x in X]
        self.Z = [z.astype(np.int32) for z in Z]

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

        Z_count = np.zeros((len(Z), len(ELEMS)), dtype=np.int64)

        for ind_mol, Z_mol in enumerate(Z):
            for ind_elem, elem in enumerate(ELEMS):
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

    def make_symmetry_functions(Z, R):
        """ Radial atom-centered symmetry functions

            Args:
                Z (np.ndarray, int): the nuclear charges of atoms in a single molecule.
                    Z has dimension [n_atom].
                R (np.ndarray, float): the raw molecular coordinates of atoms in a single molecule.
                    R is in units of angstrom and has dimension [n_atom, 3].

            Returns:
                X (np.ndarray, float): the radial symmetry functions of the molecule.
                    X is unitless and has dimension [n_atom, n_feature]. 
                    
        """

        # the number of atoms in this molecule
        natom = Z.shape[0]

        # X: the radial atom-centered symmetry functions for this molecule
        X = np.zeros((natom, len(ELEMS), len(SHIFTS)))

        # populate X
        for atom_ind_i in range(natom):

            r_i = R[atom_ind_i]

            for atom_ind_j in range(natom):

                r_j = R[atom_ind_j]
                r_ij = np.linalg.norm(r_i - r_j)

                elem_j = Z[atom_ind_j]
                elem_ind_j = ELEMS.index(elem_j)

                for shift_ind, shift in enumerate(SHIFTS):

                    shift_r_ij_sq = (r_ij - shift) ** 2

                    # edit this line!
                    #X[atom_ind_i, elem_ind_j, shift_ind] += 0.0
                    X[atom_ind_i, elem_ind_j, shift_ind] += np.exp(-WIDTH * shift_r_ij_sq)

        X = X.reshape(natom, -1)

        return X

    def normalize_features(X, Z):
        """
        normalize radial symmetry functions
        """

        X_all = np.concatenate(X)
        Z_all = np.concatenate(Z)

        means, stds = [], []

        for elem in ELEMS:
            X_elem = X_all[Z_all == elem]
            means.append(np.average(X_elem, axis=0))
            stds.append(np.std(X_elem, axis=0))

        for x, z in zip(X, Z):
            for atom_ind, atom_elem in enumerate(z):

                mean = means[ELEMS.index(atom_elem)]
                std = stds[ELEMS.index(atom_elem)]

                x[atom_ind] -= mean
                x[atom_ind] /= (std + 1e-2)

        return X

    def flatten_mols(batch):
        """ Utility function to prepare a batch of molecules for inference

            Args: batch (list of dataset items)

            Returns: 
                Xz, Iz, E, nmol

        """
    
        nmol = len(batch)
        X_all = np.concatenate([b[0] for b in batch])
        Z_all = np.concatenate([b[1] for b in batch])
        I_all = np.concatenate([np.full(len(b[0]), bi) for bi, b in enumerate(batch)])
    
        E = torch.from_numpy(np.array([b[2] for b in batch]).astype(np.float32))
    
        Xz = {}
        Iz = {}
    
        for z in ELEMS:
            mask = (Z_all == z)
            Xz[z] = torch.from_numpy(X_all[mask])
            Iz[z] = torch.from_numpy(I_all[mask])
    
        return Xz, Iz, E, nmol

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx], self.y[idx]

    def num_features(self):
        return self.n_feat


if __name__ == "__main__":

    # load (and generate features for) 100 random QM9 molecules 
    dataset = QM9Dataset("QM9.npz", 500)

    # get the per-atom feature vectors of the first molecule
    X_mol0 = dataset[0][0]
    print(f"\nX_mol0 shape: {X_mol0.shape}")

    # we'll sum the per-atom feature vectors as a correctness test
    sum_X_mol0 = np.sum(X_mol0)
    print(f"computed sum of X_mol0:  {sum_X_mol0:.6f}")

    # zach's precomputed sum of the feature vectors
    ref_sum_X_mol0 = 1002.054321
    print(f"reference sum of X_mol0: {ref_sum_X_mol0:.6f}")

    # compare your feature vector sum to zach's reference
    if np.abs(ref_sum_X_mol0 - sum_X_mol0) > 1e-3:
        raise AssertionError("Error: symmetry function implementation doesn't match reference implementation.")
    else:
        print("\nSymmetry function implementation is correct!")

