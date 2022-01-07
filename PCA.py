import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
For convenience I used library pandas to load data.
Function load_data() downloads dataset from provided link in assigment.
Extract features and store in numpy array.
As dataset values are provided in two columns with numpy hstack function
we are just joining those two rows in to the one.
"""


def load_data():
    dataset_url = 'http://lib.stat.cmu.edu/datasets/boston'
    raw_df = pd.read_csv(dataset_url, sep="\s+", skiprows=22, header=None)
    return np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]), np.asarray(raw_df.values[1::2, 2:3],
                                                                                   dtype=np.float64)


def myNormalization(X):
    X_mean = np.mean(X.T, axis=1)
    X_std = np.std(X.T, axis=1)
    return (X - X_mean) / X_std


def myPCA(X, k):
    sigma = np.cov(X.T)
    eigenvalues, eigenvectors = np.linalg.eig(sigma)

    eigenvectors = eigenvectors.T
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]
    k_components = eigenvectors[0: k]
    return np.dot(X, k_components.T)


if __name__ == '__main__':
    features, target = load_data()
    norm_features = myNormalization(features)
    X_transformed = myPCA(norm_features, 2)
    print(X_transformed.shape)

    X1 = X_transformed[:, 0]
    X2 = X_transformed[:, 1]


