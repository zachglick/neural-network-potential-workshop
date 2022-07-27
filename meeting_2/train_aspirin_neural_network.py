import numpy as np
import scipy.spatial

import torch
from torch.utils.data import Dataset, DataLoader, Subset

class AspirinDataset(Dataset):
    """Aspirin subset of MD17"""

    def __init__(self, npz_file: str, size: int):
        """
        Args:
            npz_file (string): Path to the npz_file.
            size (int): Number of molecules to load (up to 10,000)
        """

        data = np.load(npz_file)

        # R (raw molecular coordinate, Ang) : (n_mol, n_atom, 3)
        # E (energies, kcal / mol)          : (n_mol, 1)
        R = data['coords'][:size]
        E = data['energies'][:size].reshape(-1,1)

        n_mol = E.shape[0]
        n_atom = R.shape[1]

        # X (features, flatted lower triangle of distance matrix, Ang) : (n_mol, n_atom choose 2)
        X = np.zeros((n_mol, (n_atom * n_atom - n_atom) // 2), dtype=np.float64)
        indices = np.tril_indices(n_atom, -1)

        for mol_i in range(n_mol):

            # calculate the distance matrix for the molecule
            D = scipy.spatial.distance_matrix(R[mol_i], R[mol_i])

            # extract the flattened lower triangle of the distance matrix
            X[mol_i] = D[indices]

        # normalize the features and labels
        X -= np.average(X, axis=0)
        X /= np.std(X, axis=0)
        E -= np.mean(E)

        self.E = torch.from_numpy(E.astype(np.float32))
        self.X = torch.from_numpy(X.astype(np.float32))

    def __len__(self):
        return len(self.E)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.X[idx], self.E[idx])

        return sample

def train_loop(dataloader, model, loss_fn, optimizer):
    """ optimize the model predictions over a single epoch

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

def test_loop(dataloader, model, loss_fn):
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
    test_loss = 0.0

    # Compute prediction and loss; no backpropagation
    with torch.no_grad():
        for X, y in dataloader:
            y_hat = model(X)
            test_loss += loss_fn(y_hat, y).item()

    test_loss /= num_batches
    return np.sqrt(test_loss)

class NeuralNetwork(torch.nn.Module):
    """ pytorch dense feed-forward neural network"""

    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
        )

    def forward(self, x):
        """forward pass"""
        return self.linear_relu_stack(x)


if __name__ == "__main__":

    # some training hyperparameters
    learning_rate = 1e-3
    batch_size = 100
    epochs = 50

    # load the (molecule, energy) data using a custom pytorch class
    dataset = AspirinDataset("rmd17_aspirin_edited.npz", 10000)

    # length of the feature vector to describe a molecule
    N_features = dataset.X.shape[1]

    # split the dataset into training and testing subsets (80/20 split)
    N = len(dataset)
    N_train = int(0.8 * N)
    N_test = N - N_train

    dataset_train = Subset(dataset, np.arange(0, N_train))
    dataset_test = Subset(dataset, np.arange(N_train, N_train + N_test))

    dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize a neural network model object to predict energies from molecular features
    model = NeuralNetwork(N_features)

    # initialize an optimizer to adjust model weights (minimizing loss) via backpropagation
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # compute the RMSE on the train and test subsets
    rmse_train = test_loop(dataloader_train, model, loss_fn)
    rmse_test = test_loop(dataloader_test, model, loss_fn)
    print("                                   Train    Test")
    print("                                   -----    -----")
    print(f"Pre-training RMSE (kcal / mol): {rmse_train:>8.3f} {rmse_test:>8.3f}")

    # optimize the model and continuously recalculate the train / test RMSE
    for t in range(epochs):
        rmse_train = train_loop(dataloader_train, model, loss_fn, optimizer)
        rmse_test = test_loop(dataloader_test, model, loss_fn)
        print(f"Epoch {t+1:5d}  RMSE (kcal / mol): {rmse_train:>8.3f} {rmse_test:>8.3f}")
