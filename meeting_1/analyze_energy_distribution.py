import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv("QM9.csv")
    energies = df["E (Hartree)"].to_numpy()
