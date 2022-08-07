# Practical Introduction to Neural Network Potentials
## Week 3 : Behler-Parrinello Neural Networks

This repository contains 4 files:

* `QM9.npz` : The same [QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904) dataset from the first meeting. Unlike the first meeting, atomic coordinates are also included.
* `qm9_dataset.py` Contains the `QM9Dataset` class for easily loading and managing the geometries and energies in `QM9.npz`.
This file is also where you finish implementing radial symmetry functions (i.e. a constant length vector to describe the local environment of a single atom in a molecule).
* `behler_parrinello_neural_network.py` Contains the `BehlerParrinelloNeuralNetwork` class, which is a neural network object for modeling transferable potential energy surfaces of various molecules. This class consists of five independent feed-forward networks, one for predicting atomic energies of H, C, N, O, and F. This file is where you implement a neural network architecture (i.e. choose a sequence of various size layers and activation functions).
* `train_qm9_neural_network.py` The main driver class that trains a `BehlerParrinelloNeuralNetwork` object using the `QM9Dataset` data. You'll edit the hyperparameters such as learning rate, batch size, number of training epochs, and amount of training data.

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

Run `train_qm9_neural_network.py` to train a BPNN on QM9.
How low of a RMSE do you get using 1,000 training molecules for 300 epochs?
Repeat the exercise with greater numbers of training molecules to generate a learning curve.
With additional training molecules, you should train for fewer epochs, so that the total number of back-propagations is approximately consistent.
Use these values:

| Molecules | Epochs |
| --------- | ------ |
| 1,000     | 300    |
| 2,000     | 200    |
| 5,000     | 150    |
| 10,000    | 100    |
| 20,000    | 50     |
| 50,000    | 30     |
| 100,000   | 15     |
| 131,885   | 15     |

Plot the validation RMSE against the number of molecules.
