# Practical Introduction to Neural Network Potentials
## Week 3 : Behler-Parrinello Neural Networks

This repository contains X files:

* `QM9.npz`
* `qm9_dataset.py`
* `behler_parrinello_neural_network.py`
* `train_qm9_neural_network.py`

### First Exercise: Implement radial symmetry functions efficiently

Run the file `qm9_dataset.py` from the command line.
When this file is executed, the QM9 dataset is loaded, features are computed for each atom in each molecule, and the features are checked for correctness.
The symmetry functions have not been fully implemented.
Verify that the output indicates that the symmetry functions are not (yet) correct.

Next, open `qm9_dataset.py` and go to the function `make_symmetry_functions.py`.
To finish implementing this function, you have to edit a single line.
This line in indicated by a comment.
Re-run `qm9_dataset.py` to determine if your changes are correct.

Once you've correctly implemented the symmetry functions, make note of the amount of time required to compute the symmetry functions.
This is printed out when running `qm9_dataset.py`

Next, replace the entire contents of `make_symmetry_functions` with the following code:
```
        natom = Z.shape[0]
        Z_onehot = np.zeros((natom, len(ELEMS)), np.int64)
        for zi, elem in enumerate(Z):
            Z_onehot[zi, ELEMS.index(elem)] = 1

        D = scipy.spatial.distance_matrix(R, R)
        X = D.reshape(natom, natom, 1) - SHIFTS.reshape(1, 1, -1)
        X = np.exp(-WIDTH* np.square(X))
        X = np.einsum("ijr,jz->izr", X, Z_onehot)
        X = X.reshape(natom, -1)

        return X
```

Verify that this different implementation is also correct by re-running `qm9_dataset.py` from the command line.
Note how the time required to compute the symmetry functions is reduced.

### Second Exercise: Train a BPNN model on QM9

Run `train_qm9_neural_network.py`
