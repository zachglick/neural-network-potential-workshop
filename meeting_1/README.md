# Practical Introduction to Neural Network Potentials
## Week 1 : Potential Energy Surfaces and Linear Regression

This repository contains three files:

* `QM9.csv` A compressed representation of [QM9](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904),
a popular dataset of 133,885 stable organic molecules with up to nine heavy (non-hydrogen) atoms, [C, O, N, F]. QM9 is often used to benchmark ML potentials.
  - The file contains 133,885+1 lines (one per molecule plus a header) and seven columns
  - The first column is the molecule id
  - The next five columns are the element counts of the five elements present in QM9 (HCONF)
  - The seventh and final column is the optimized ground state energy of the molecule calculated at the B3LYP/6-31G(2df,p) level of theory

* `analyze_energy_distribution.py` A (mostly empty) python file for the first exercise.

* `run_linear_regression.py` A (mostly empty) python file for the second exercise.

---

### First Exercise : `analyze_energy_distribution.py`

This exercise is intended to get you thinking about the target regression variable: the molecular ground state energy.

Plot a histogram of the QM9 energy distribution and print out some values associated with this distribution: minimum, maximum, range, average, and standard deviation, etc.
What unit are the energies in?
Are all energies the same sign?
Why or why not?

The average covalent bond is on the order of 100 kcal / mol.
Convert this energy to the same units as the QM9 energies.
What is 100 kcal / mol error as a fraction of the standard deviation of energies?
For a ML model to accurately capture bond-breaking, it must have an error much less than the average bond dissociation energy.
Does that seem feasible? How accurate must the ML model be?

---

### Second Exercise : `run_linear_regression.py`

This exercise uses the scikit-learn library to assess the performance of a simple linear regression model.

Print out and examine the numpy arrays $X$ and $y$. Ensure that you understand the meaning of the values. Print out the shapes of both arrays and make sure you understand the meaning of each dimension. How many parameters does linear regression require for this data?

Read the scikit-learn documentation for [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
Use the (already instantiated) LinearRegression object to fit the molecular atom counts ($X$) to the energies ($y$).
Next, evaluate the LinearRegression model on the same data to get $\hat{y}$.
What's the mean absolute error (MAE) of these predictions? What about the root mean squared error (RMSE)?
How do these numbers compare to the strength of a covalent bond? What about a strong intermolecular bond, such as a hydrogen bond?

Print out the value of the linear regression parameters (the weights and the bias).
What are the units of these parameters?
Intuitively, what do they mean? 
How do they relate to atomic number?

If we added molecules containing Boron or Chlorine to the dataset and refit the model, what values do you think the two new parameters would approximately take?

Do you think your regression model would overestimate or underestimate the energy of isolated atoms?

Make a scatter plot of $y$ by $\hat{y}$ and also of $y$ by $\hat{y} - y$.

Repeat this exercise without an intercept. Do the results change? Is that expected?
