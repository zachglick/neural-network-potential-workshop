# Practical Introduction to Neural Network Potentials
## Week 2 : Fitting the Potential Energy Surface of aspirin with a Neural Network

This repository contains four files:

* `rmd17_aspirin_edited.npz` A subset of [revised MD17](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038),
a popular dataset of 100,000 molecular dynamics geometries (with DFT energies and forces) for ten different molecules.
  - The file contains 100,000 geometries and energies of only the aspirin molecule
  - The file can be read with `numpy.loadz`
  - Geometries are stored in Angstrom; energies in kcal / mol

* `aspirin_dataset.py` Contains the `AspirinDataset` class for easily loading and managing the geometries and energies in `rmd17_aspirin_edited.npz`.
This file is also where you implement a molecular feature vector (i.e. a constant length vector to describe an arbitrary geometry of the aspirin molecule).

* `pes_neural_network.py` Contains the `PESNeuralNetwork` class, which is a neural network object for modeling the potential energy surface of a single molecule (such as aspirin). This file is where you implement a neural network architecture (i.e. choose a sequence of various size layers and activation functions).

* `train_aspirin_neural_network.py` The main driver class that trains a `PESNeuralNetwork` object using the `AspirinDataset` data. You'll edit the hyperparameters such as learning rate, batch size, number of training epochs, and amount of training data.

---

### First Exercise : Train a simple model

This exercise is intended to get you familiar with the process of training a neural network

Run the file `train_aspirin_neural_network.py` from the command line.
Observe the training and validation errors each epoch.
Make a note of the best validation error over 100 epochs.
Does the model seem to be learning anything?
Do you think this model is converged?

---

### Second Exercise : Tune the learning rate

This exercise is intended to get you familiar with the process of training a neural network

Open the file `train_aspirin_neural_network.py` and change the value of the `learning_rate` variable to a larger number.
Repeat the first exercise.
Do you think your new learning rate is better or worse?
Repeat this exercise with learning rates between $10^{-1}$ and $10^{-7}$, making sure to try the extreme values.
Can you explain the behavior at the extreme values?
What's the best learning rate you can find?
Leave the `learning_rate` variable set to your best hand-tuned value.

---

### Third Exercise : Choose an architecture

This exercise is intended to get you familiar with the process of constructing neural network architectures.

Open the file `pes_neural_network.py`.
Read about constructing neural networks with pytorch [here](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html).
Also look [here](https://pytorch.org/docs/stable/nn.functional.html) for a full list of possible layers and activation functions.

Sketch the current architecture on a piece of paper (or white board).
Can you count the number of parameters in this model?
This architecture is intentionally not optimized.
Edit the architecture, adding one or more hidden layers (and activation functions) and also increase the dimensionality of one or more hidden layers.
Make sure that the final output is dimension 1, and also that the output of each hidden layer matches the input of the next hidden layer.

Re-run `train_aspirin_neural_network.py`.
Observe how the amount of time per epoch changed.
Did your new architecture result in a better validation error?
If not, continue to modify the architecture until you can achieve a better validation error.
If you do get a better validation error, see if you can improve it further.
You make have to re-tune the learning rate.

---

### Fourth Exercise : Choose a feature vector

This exercise is intended to get you thining about how to featurize a molecular system.

Open the file `aspirin_dataset.py`. You can read about the Dataset class [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).
Find the following line:
```
        X = AspirinDataset.make_features_raw_coords(R)
```
This function computes a constant-length per-molecule feature vector (`X`) from the raw atomic coordinates (`R`).
In this case, the raw coordinates are used as the feature vector without modification, so the function doesn't do much.

Raw coordinates may not be the best choice for a feature vector.
Imagine two identical aspirin molecules, one a translated and/or rotated copy of another.
Should they have the same energy?
Will they have the same energy according to a neural network using this feature vector?

Comment out the raw coordinate line and uncomment the subsequent line:
```
        #X = AspirinDataset.make_features_distance_matrix(R)
```
This function assigns interatomic distances (instead of raw coordinates) to the constant-length per-molecule feature vector (`X`).
Are distances translationally and rotationally invariant?

Re-run `train_aspirin_neural_network.py` and see if you get better results (i.e. a lower validation RMSE).
You make have to re-tune the learning rate.

---

### Fifth Exercise : Optimize the training process

After completing the first four excercises, see how low you can get the validation RMSE.

You may want to try implementing your own custom feature vector. You can do this by implementing the `make_features_distance_custom` function in `aspirin_dataset.py`.
Consider using angles in addition to distances.

Besides tweaking the learning rate and architecture, other tunable (hyper)parameters are the batch size, the amount of training data used, and number of epochs trained. 
The optimal learning rate is particularly sensitive to batch size.
These values are specified at the top of the `__main__` function in `train_aspirin_neural_network.py`.
