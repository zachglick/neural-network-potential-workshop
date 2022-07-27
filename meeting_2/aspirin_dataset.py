import numpy as np
import scipy.spatial
import torch

class AspirinDataset(torch.utils.data.Dataset):
    """pytorch Dataset class for handling Aspirin subset of MD17"""

    def make_features_raw_coords(R):
        """ Get the simplest possible feature vector

            Args:
                R (numpy array): the raw molecular coordinates of all aspirin molecules.
                    R has dimension [n_mol, n_atom, 3]

            Returns:
                R_flat (numpy array): The input coordinates with the last two indices flattened.
                                      R_flat has dimension [n_mol, n_atom * 3]
        """

        n_mol = R.shape[0]
        n_atom = R.shape[1]

        R_flat = R.reshape((n_mol, n_atom * 3))

        return R_flat

    def make_features_distance_matrix(R):
        """ A feature vector of distances between all unique atom pairs in the molecule

            Args:
                R (numpy array): the raw molecular coordinates of all aspirin molecules.
                    R has dimension [n_mol, n_atom, 3]

            Returns:
                D_lower (numpy array): distances between all unique pairs of atoms
                    D_lower has dimension [n_mol, 0.5 * (n_atom) * (n_atom - 1) ]
        """

        n_mol = R.shape[0]
        n_atom = R.shape[1]
        n_dist = (n_atom * (n_atom - 1)) // 2 

        # flattened, lower triangle of the distance matrix 
        # lower triangle (not whole matrix) has all unique distances
        D_lower = np.zeros((n_mol, n_dist), dtype=np.float64)
        lower_triangular_indices = np.tril_indices(n_atom, -1)

        for mol_i in range(n_mol):

            # calculate the distance matrix for this molecule (n_atom * n_atom)
            D = scipy.spatial.distance_matrix(R[mol_i], R[mol_i])

            # extract the flattened lower triangle of this distance matrix
            D_lower[mol_i] = D[lower_triangular_indices]

        return D_lower

    def make_features_custom(R):
        """ Your own feature vector

            Args:
                R (numpy array): the raw molecular coordinates of all aspirin molecules.
                    R has dimension [n_mol, n_atom, 3]

            Returns:
        """

        # code your own features here
        return 

    def __init__(self, npz_file: str, size: int):
        """
        Args:
            npz_file (string): Path to the npz_file.
            size (int): Number of molecules to load (up to 100,000)
        """

        assert 2000 < size <= 100000

        # load the raw data from saved numpy file
        data = np.load(npz_file)

        # R (raw molecular coordinate, Ang) : (n_mol, n_atom, 3)
        # E (energies, kcal / mol)          : (n_mol, 1)

        inds = np.arange(98000)
        np.random.shuffle(inds)
        inds = np.concatenate((inds[:size-2000], np.arange(98000, 100000)))

        R = data['coords'][inds]
        E = data['energies'][inds].reshape(-1,1)

        X = AspirinDataset.make_features_raw_coords(R)
        #X = AspirinDataset.make_features_distance_matrix(R)
        #X = AspirinDataset.make_features_custom(R)
        y = E

        # normalize the features (X) and labels (y)
        X -= np.average(X, axis=0)
        X /= np.std(X, axis=0)
        y -= np.mean(y)

        # convert X and y into pytorch tensors (similar to numpy arrays)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.X = torch.from_numpy(X.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]

