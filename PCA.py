""""
Dependencies:
- numpy (calculation stuff)
- pandas (just for convenience of reading data. This way it is much faster)
- matplotlib (plot stuff)

If You have all dependency, script needs to be just run. Data will be automatically download and graph will be displayed

CONCLUSION:
House prices that are lower or equal to 15 and prices that are more or equal to 30 are nicely separated.
By using first two principal components we can capture approx. 58% variance meaning they are adding majority
of information value about data.
"""""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

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


"""
Standard process of normalizing data. First find mean then find standard deviation.
Final step subtract mean from data and divide by standard deviation.    
"""
def myNormalization(X):
    X_mean = np.mean(X.T, axis=1)
    X_std = np.std(X.T, axis=1)
    return (X - X_mean) / X_std


"""
Calculating variance, so we can see how much information  which pca component produce
"""
def _calculate_variance(eigenvalues):
    variance_for_each_comp = []
    total_sum = sum(eigenvalues)
    for i in sorted(eigenvalues, reverse=True):
        variance_for_each_comp.append((i / total_sum) * 100)
    cumulative_sum_variance_for_each_comp = np.cumsum(variance_for_each_comp)
    print("\nVariance capcured by each component \n", variance_for_each_comp)
    print("\nCumulative sum of variance  \n", cumulative_sum_variance_for_each_comp)


"""
As input to the function myPCA are normalized features.
1st calculate covariance matrix
2nd calculate eigenvalues anc eigenvectors
3rd sort eigenvectors. In descent order
4th transform data to k components
"""
def myPCA(X, k):
    # calculate covariance matrix
    sigma = np.cov(X.T)

    # calculate eigenvalues anc eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(sigma)
    _calculate_variance(eigenvalues)

    # sort eigenvectors
    eigenvectors = eigenvectors.T
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]

    # transform data to k components
    k_components = eigenvectors[0: k]
    return np.dot(X, k_components.T)



if __name__ == '__main__':
    features, target = load_data()
    norm_features = myNormalization(features)
    X_transformed = myPCA(norm_features, 2)

    # Get indices that meet define criteria in assignment for plotting data.
    indices_cheap = np.nonzero(target <= 15.0)
    indices_expensive = np.nonzero(target >= 30.0)
    indices_cheap = np.reshape(indices_cheap, newshape=(-1))
    indices_expensive = np.reshape(indices_expensive, newshape=(-1))

    # Filtering the house prices by condition price.
    house_prices_cheap = target[indices_cheap]
    house_prices_expensive = target[indices_expensive]

    X_transform_plot_data_cheap = X_transformed[indices_cheap]
    X_transform_plot_data_expensive = X_transformed[indices_expensive]

    x1_cheap = X_transform_plot_data_cheap[:, 0]
    x2_cheap = X_transform_plot_data_cheap[:, 1]

    x1_expensive = X_transform_plot_data_expensive[:, 0]
    x2_expensive = X_transform_plot_data_expensive[:, 1]

    # setting up plot
    plt.scatter(x1_cheap, x2_cheap, c=house_prices_cheap, edgecolor="none", alpha=0.8,
                cmap=plt.cm.get_cmap("Purples", 1))
    plt.scatter(
        x1_expensive, x2_expensive, c=house_prices_expensive, edgecolor="none", alpha=0.8,
        cmap=plt.cm.get_cmap("Greens", 1)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(handles=[mpatches.Patch(facecolor='purple', edgecolor='purple', label='under 15'),
                        mpatches.Patch(facecolor='green', edgecolor='green', label='above 30')])
    plt.show()
