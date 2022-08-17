import time
import numpy as np

import torch
from torch.utils.data import Subset

import torch_geometric
from torch_geometric.loader import DataLoader

from qm9_dataset import QM9Dataset
from message_passing_neural_network import MessagePassingNeuralNetwork

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
    model.train()

    #for batch, (X, I, y, nmol) in enumerate(dataloader):
    for batch in dataloader:

        y = batch.y

        # Compute prediction and loss
        y_hat = model.forward(batch)
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
    model.eval()

    # Compute prediction and loss; no backpropagation
    with torch.no_grad():
        for batch in dataloader:
            y = batch.y
            y_hat = model.forward(batch)
            val_loss += loss_fn(y_hat, y).item()


    val_loss /= num_batches
    return np.sqrt(val_loss)

if __name__ == "__main__":

    # training hyperparameters: edit these!
    learning_rate = 1e-3
    batch_size = 100
    epochs = 100
    N_train = 5000 # keep this in range [1, 131885]

    distance_cutoff = 3.0
    embedding_dim = 8
    message_passes = 0

    ###########################################################################

    # we'll always validate on the last 2000 molecules; don't change this
    N_val = 2000

    # load the (molecule, energy) data using a custom pytorch class
    dataset = QM9Dataset("../meeting_3/QM9.npz",
                         size=(N_train + N_val),
                         elems=[1, 6, 7, 8, 9],
                         distance_cutoff=distance_cutoff)

    dataset_train = Subset(dataset, np.arange(0, N_train))
    dataset_val = Subset(dataset, np.arange(N_train, N_train + N_val))

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)#, collate_fn=dataset.flatten_mols)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)#, collate_fn=dataset.flatten_mols)

    # initialize a neural network model object to predict energies from molecular features
    model = MessagePassingNeuralNetwork(
                atoms=[1, 6, 7, 8, 9],
                distance_cutoff=distance_cutoff,
                embedding_dim=embedding_dim,
                npass=message_passes,
            )
    #model.load_state_dict(torch.load("model.pt"))

    # initialize an optimizer to adjust model weights (minimizing loss) via backpropagation
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # compute the RMSE on the train and val subsets
    rmse_train = val_loop(dataloader_train, model, loss_fn)
    rmse_val = val_loop(dataloader_val, model, loss_fn)
    print()
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
        if rmse_val < rmse_val_lowest:
            torch.save(model.state_dict(), "model.pt")
            rmse_val_lowest = rmse_val
            best_epoch = "*"
        else:
            best_epoch = " "

        print(f"Epoch {t+1:4d} ({d_time:<4.1f} sec)     RMSE: {rmse_train:>8.3f} {rmse_val:>8.3f}  {best_epoch}")


    #model2 = MessagePassingNeuralNetwork([1, 6, 7, 8, 9])
    #model2.load_state_dict(torch.load("model.pt"))
    #model2.eval()
