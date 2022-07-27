import time
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from aspirin_dataset import AspirinDataset
from pes_neural_network import PESNeuralNetwork

def train_loop(dataloader, model, loss_fn, optimizer):
    """ optimize the model weights over entire dataset (one epoch)

        Args:
            dataloader: pytorch dataloader object
            model: pytorch neural network object
            loss_fn: pytorch function for computing model loss
            optimizer: pytorch optimizer for minimizing loss

        Returns:
            RMSE over all data
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):

        # Compute prediction and loss
        y_hat = model(X)
        loss = loss_fn(y_hat, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss

    train_loss /= num_batches
    return np.sqrt(train_loss.detach().numpy())

def val_loop(dataloader, model, loss_fn):
    """ compute model predictions on an entire dataset

        Args:
            dataloader: pytorch dataloader object
            model: pytorch neural network object
            loss_fn: pytorch function for computing model loss

        Returns:
            RMSE over all data
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss = 0.0

    # Compute prediction and loss; no backpropagation
    with torch.no_grad():
        for X, y in dataloader:
            y_hat = model(X)
            val_loss += loss_fn(y_hat, y).item()

    val_loss /= num_batches
    return np.sqrt(val_loss)

if __name__ == "__main__":

    # training hyperparameters: edit these!
    learning_rate = 1e-5
    batch_size = 100
    epochs = 100
    N_train = 100 # keep this in range [1, 98000]

    ###########################################################################

    # we'll always validatie on the last 2000 aspirin molecules; don't change this
    N_val = 2000

    # load the (molecule, energy) data using a custom pytorch class
    dataset = AspirinDataset("rmd17_aspirin_edited.npz", N_train + N_val)

    # length of the molecular feature vector
    N_features = dataset.X.shape[1]

    dataset_train = Subset(dataset, np.arange(0, N_train))
    dataset_val = Subset(dataset, np.arange(N_train, N_train + N_val))

    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize a neural network model object to predict energies from molecular features
    model = PESNeuralNetwork(N_features)

    # initialize an optimizer to adjust model weights (minimizing loss) via backpropagation
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # compute the RMSE on the train and val subsets
    rmse_train = val_loop(dataloader_train, model, loss_fn)
    rmse_val = val_loop(dataloader_val, model, loss_fn)
    print("                                   Train     Val")
    print("                                    (kcal / mol)")
    print("                                   --------------")
    print(f"Pre-training              RMSE: {rmse_train:>8.3f} {rmse_val:>8.3f}")

    # record the lowest val rmse throughout training
    rmse_val_lowest = rmse_val

    # optimize the model and continuously recalculate the train / val RMSE
    for t in range(epochs):
        time_start = time.time()
        rmse_train = train_loop(dataloader_train, model, loss_fn, optimizer)
        rmse_val = val_loop(dataloader_val, model, loss_fn)
        d_time = time.time() - time_start

        # is the val RMSE at a new low?
        best_epoch = "*" if rmse_val < rmse_val_lowest else " "
        rmse_val_lowest = min(rmse_val, rmse_val_lowest)

        print(f"Epoch {t+1:4d} ({d_time:<4.1f} sec)     RMSE: {rmse_train:>8.3f} {rmse_val:>8.3f}  {best_epoch}")
