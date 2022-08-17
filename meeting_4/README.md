# Practical Introduction to Neural Network Potentials
## Week 4 : Message-Passing Neural Networks (MPNNs)

This repository contains 3 files:

* `qm9_dataset.py` Contains the `QM9Dataset` class for easily loading and managing the geometries and energies in `QM9.npz`.
* `message_passing_neural_network.py` Contains the `MessagePassingNeuralNetwork` class, which is a neural network object for modeling transferable potential energy surfaces of various molecules. This class defines message, update, and readout functions that operate on molecular graphs for predicting atomic energies. You will experiment with different message, update, and readout functions.
* `train_qm9_neural_network.py` The main driver class that trains a `MessagePassingNeuralNetwork` object using the `QM9Dataset` data. You'll edit the parameters and hyperparameters such as learning rate, batch size, number of training epochs, embedding dimension, and number of message passes.

### First Exercise: Experiment with the number of message passes

Run the file `train_qm9_neural_network.py` to train a MPNN on 5000 QM9 molecules for 100 epochs.
Make note of the best validation RMSE and time per epoch.

Open the file and increment the `message_passes` variable.
Do you see a change in best validation RMSE and time per epoch?

Repeat this experiment two more times, setting the `message_passes` variable to 3 and 5.
Of the four runs, which is the best?
Do more message passing always improve performance?

### Second Exercise: Implement a better message function

Open the file `message_passing_neural_network.py` and find the `forward` function, which defines a 
forward pass of a batch of molecules through the network.
Within this function, find where the messages are defined

The current message function (using the notation from the slides) is defined as:

$M_{i}^{t} = \sum_{j} m_{ji}^{t} = \sum_{j} (x_{j}^{t-1}, e_{ij})$

where $x_{i}^{t}$ is the hidden state vector of atom $i$ at iteration $t$, $e_{ij}$ is an encoding of the scalar distance between atoms $i$ and $j$, and $(... , ...)$ is vector concatenation.

Your task is to come up with a better message function.
One possible improvement is to replace the concatenation operation with an outer product between vectors. (Hint: you'll use the `torch.einsum` function).
You could also introduce an additional feed-forward neural network (the update and readout functions already use neural networks). There are many good alternatives.

Note: the dimensionality of the message vector has to be defined in the __init__ function.
For the current message function, the vector length is the sum of the hidden state vector length and the edge vector length:
```
self.message_dim = nshifts + self.embedding_dim
```
When you define your new message function, you'll have to change this line to correspond to the new dimensionality of the message vector.


### Third Exercise: Experiment with the distance cutoff and embedding dimension

The `embedding_dim` parameter controls the size of the hidden state vector.
Train a MPNN with both a smaller and larger embedding dimension, and make note of the effect on validation RMSE and time per epoch.

The `distance_cutoff` parameter is the maximum distance of an edge in the molecular graph.
Again, train a MPNN with both a smaller and larger distance cutoff, and make note of the effect on validation RMSE and time per epoch.

Optional: repeat the first exercise with your improved message function.
Did your conclusions about the number of message passes change?

### Fourth Exercise: Compare your MPNN to last week's BPNN

Recreate the learning curve from last week with an MPNN.
Train this MPNN with the number of message passes, the message passing function, the distance cutoff, and the embedding dimension informed by the results of the previous exercises.

Which model performs better?
Note that model performance is heavily dependent on your chosen hyperparameters and also achieving convergence (not terminating the training too early).
