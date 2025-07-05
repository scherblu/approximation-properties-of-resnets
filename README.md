# Approximation Properties of Residual Neural Networks

by Lucas Scherberger

This repository contains the code of the Bachelor's thesis "Approximation Properties of Residual Neural Networks", supervised by JProf. Diyora Salimova at the University of Freiburg in 2025.

## Installation
See environment.yml for recreating a conda environment. 

## Code structure
The function being approximated is given by generated pairs of inputs and outputs in generate_data.py and formatted as training and test dataloaders. 
The ANN and ResNet models themselves are constructed in model.py using PyTorch.
The models are trained in train.py using the Adam optimizer and are evaluated with the test dataset after each epoch.
The complete process from generating data to training and evaluating on the test dataset is executed in main.py. 

