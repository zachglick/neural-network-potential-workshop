import time
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset

from qm9_dataset import QM9Dataset
from behler_parrinello_neural_network import BehlerParrinelloNeuralNetwork

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

    for batch, (X, I, y, nmol) in enumerate(dataloader):

        # Compute prediction and loss
        y_hat = model.forward(X, I, torch.zeros(nmol, 1))
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
        for X, I, y, nmol in dataloader:
            y_hat = model.forward(X, I, torch.zeros(nmol, 1))
            val_loss += loss_fn(y_hat, y).item()


    val_loss /= num_batches
    return np.sqrt(val_loss)

if __name__ == "__main__":

    # training hyperparameters: edit these!
    learning_rate = 5e-4
    batch_size = 100
    epochs = 300
    N_train = 1000 # keep this in range [1, 131885]
    symmetry_function_shifts = np.linspace(1.0, 4.0, 33)
    symmetry_function_width = 1.0
    do_normalize_features = True

    ###########################################################################

    # we'll always validate on the last 2000 molecules; don't change this
    N_val = 2000

    # load the (molecule, energy) data using a custom pytorch class
    dataset = QM9Dataset("QM9.npz",
                         size=(N_train + N_val),
                         elems=[1, 6, 7, 8, 9],
                         shifts=symmetry_function_shifts,
                         width=symmetry_function_width,
                         do_normalize_features=do_normalize_features,
                         )
    scale = dataset.scale

    # length of the molecular feature vector
    num_feat = dataset.num_features()

    dataset_train = Subset(dataset, np.arange(0, N_train))
    dataset_val = Subset(dataset, np.arange(N_train, N_train + N_val))

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset.flatten_mols)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=dataset.flatten_mols)

    # initialize a neural network model object to predict energies from molecular features
    model = BehlerParrinelloNeuralNetwork(num_feat, [1, 6, 7, 8, 9])

    # initialize an optimizer to adjust model weights (minimizing loss) via backpropagation
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50], gamma=0.2)

    # compute the RMSE on the train and val subsets
    #model.eval()
    rmse_train = val_loop(dataloader_train, model, loss_fn) / scale
    rmse_val = val_loop(dataloader_val, model, loss_fn) / scale
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
        rmse_train = train_loop(dataloader_train, model, loss_fn, optimizer) / scale
        #scheduler.step()
        rmse_val = val_loop(dataloader_val, model, loss_fn) / scale
        d_time = time.time() - time_start

        # is the val RMSE at a new low?
        best_epoch = "*" if rmse_val < rmse_val_lowest else " "
        rmse_val_lowest = min(rmse_val, rmse_val_lowest)

        print(f"Epoch {t+1:4d} ({d_time:<4.1f} sec)     RMSE: {rmse_train:>8.3f} {rmse_val:>8.3f}  {best_epoch}")
