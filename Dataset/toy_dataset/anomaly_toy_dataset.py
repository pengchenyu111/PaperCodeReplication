from sklearn.datasets import make_moons, make_blobs
import numpy as np

n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
rng = np.random.RandomState(42)

data1 = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, random_state=0, n_samples=n_inliers, n_features=2)
data2 = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5], random_state=0, n_samples=n_inliers, n_features=2)
data3 = make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, 0.3], random_state=0, n_samples=n_inliers, n_features=2)
data4 = 4.0 * (make_moons(n_samples=n_samples, noise=0.05, random_state=0)[0] - np.array([0.5, 0.25]))
data5 = 14.0 * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)

data1 = np.concatenate([data1[0], rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
data2 = np.concatenate([data2[0], rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
data3 = np.concatenate([data3[0], rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
data4 = np.concatenate([data4, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
data5 = np.concatenate([data5, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)
