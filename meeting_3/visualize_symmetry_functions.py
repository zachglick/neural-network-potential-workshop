import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from qm9_dataset import QM9Dataset


def load_xyz():

    elem_to_Z = {
            "H" : 1,
            "C" : 6,
            "N" : 7,
            "O" : 8,
            "F" : 9,
    }

    lines = open("QM9_200.xyz").readlines()
    natom = int(lines[0])

    lines = lines[2:2+natom]
    lines = [line.strip().split() for line in lines]

    Z = [elem_to_Z[line[0]] for line in lines]
    R = [np.array(line[1:4]).astype(np.float64) for line in lines]

    Z, R = np.array(Z), np.array(R)

    return Z, R


if __name__ == "__main__":


    # load nuclear charges and coordinates from an xyz
    Z, R = load_xyz()

    # these parameters specify the symmetry functions
    SHIFTS = np.linspace(0.8, 5.0, 43)
    WIDTH = 10.0

    # plot the symmetry functions of each atom
    fig, axs = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, squeeze=True, figsize=(12,5))
    axs = axs.ravel()

    allax = fig.add_subplot(111, frameon=False)
    allax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    allax.set_xlabel("Distance from atom ($\mathrm{\AA}$)")
    allax.set_ylabel("Symmetry Function Value")

    plt.subplots_adjust(wspace=0.0, hspace=0.0)

    X = QM9Dataset.make_symmetry_functions(Z, R, [1, 6, 7, 8, 9], SHIFTS, WIDTH) 

    def plot_data(X):

        for i in range(10):

            axs[i].clear()
            plotH = axs[i].plot(SHIFTS, X[i,0,:], label="H", marker=".", lw=1.0, ms=2.0, color="black")
            plotC = axs[i].plot(SHIFTS, X[i,1,:], label="C", marker=".", lw=1.0, ms=2.0, color="grey")
            plotN = axs[i].plot(SHIFTS, X[i,2,:], label="N", marker=".", lw=1.0, ms=2.0, color="blue")
            plotO = axs[i].plot(SHIFTS, X[i,3,:], label="O", marker=".", lw=1.0, ms=2.0, color="red")
            axs[i].set_ylim(top=4.0)

        return [plotH[0], plotC[0], plotN[0], plotO[0]]

    handles = plot_data(X)

    fig.legend(handles=handles,
               labels=["H", "C", "N", "O"],
               loc="center right",
               borderaxespad=1.0)

    # recompute symmetry functions and update the plot when the slider is changed

    ax_slider = plt.axes([0.03, 0.40, 0.02, 0.20], facecolor='lightgoldenrodyellow')
    slider_width = Slider(ax_slider,
                          label="WIDTH",
                          valmin=-4,
                          valmax=4,
                          valinit=np.log10(WIDTH),
                          valfmt="10 ^ %.2f",
                          orientation="vertical") 

    def update_slider(val):

        WIDTH = 10.0 ** val
        X = QM9Dataset.make_symmetry_functions(Z, R, [1, 6, 7, 8, 9], SHIFTS, WIDTH) 
        plot_data(X)
        fig.canvas.draw_idle()

    slider_width.on_changed(update_slider)

    plt.show()
