# Practical Introduction to Neural Network Potentials
## Week 1 : Potential Energy Surfaces and Linear Regression

This repository contains three files:

A compressed representation of (QM9)[https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904],
a dataset of 133,885 stable organic molecules with up to nine heavy atoms (CONF). The file contains 133,885 lines (one per molecule) and seven columns.

A python file titled analyze_energy_distribution.py for

A python file titled run_linear_regression.py for



### Exercises : analyze_energy_distribution.py

These exercises are intended to get you thinking about the target regression variable: the molecular ground state energy.

Plot a histogram of the QM9 energy distribution.
and print out some values associated with this distribution: minimum, maximum, range, average, and standard deviation, etc.
What unit are the energies in?
Are all energies the same sign?
Why or why not?

The average carbon-carbon bond dissocation energy is between 80 and 160 kcal / mol.
Convert this energy range to the same units as the QM9 energies.
For a PES regression model to accurately model bond-breaking, it must have an error much less than the average bond dissociation energy.
Does that seem feasible? What is an 80 kcal / mol error as a fraction of the distribution standard deviation

### Exercises : run_linear_regression.py

Use the scikit-learn library to fit and assess the performance of a simple linear regression model.

Print out the numpy arrays X and y. Print out their shapes. Ensure that you understand the meanings of the values and the dimensions.

Read the scikit-learn documentation for (LinearRegression)[https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html].
Use the (already instantiated) LinearRegression object to fit the molecular atom counts to the energies.
Next, evaluate the LinearRegression model on the same data to get \hat{y}.
What's the mean absolute error (MAE) of these predictions? What about the root mean squared error?

Make a scatter plot of y by yhat. This is a correlation plot.
Make a scatter plot of y by yhat - y.


Repeat the fitting without the intercept. Do the results change? Is that expected?
