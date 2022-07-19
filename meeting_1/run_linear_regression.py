import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    df = pd.read_csv("QM9.csv")
    
    X = df[["n_H", "n_C", "n_N", "n_O", "n_F"]].to_numpy()
    y = df[["E (Hartree)"]].to_numpy()
    model = LinearRegression(fit_intercept=True)
