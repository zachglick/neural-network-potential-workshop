import numpy as np
import matplotlib.pyplot as plt

from qm9_dataset import QM9Dataset


elem_to_Z = {
        "H" : 1,
        "C" : 6,
        "N" : 7,
        "O" : 8,
        "F" : 9,
        }


if __name__ == "__main__":


    lines = open("QM9_200.xyz").readlines()
    natom = int(lines[0])

    lines = lines[2:2+natom]
    lines = [line.strip().split() for line in lines]

    Z = [elem_to_Z[line[0]] for line in lines]
    R = [np.array(line[1:4]).astype(np.float64) for line in lines]

    Z, R = np.array(Z), np.array(R)

    X = QM9Dataset.make_symmetry_functions(Z, R, [1, 6, 7, 8, 9], np.linspace(1.0, 5.0, 33), 1.0) 

    print(Z)
    print(R)
    print(X)






