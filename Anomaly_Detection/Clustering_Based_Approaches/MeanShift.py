"""
    均值漂移聚类

    1、参考文章
        https://blog.csdn.net/weixin_42056745/article/details/101287231
    2、原理：
        1. 确定滑动窗口半径r，以随机选取的中心点C半径为r的圆形滑动窗口开始滑动。均值漂移类似一种爬山算法，在每一次迭代中向密度更高的区域移动，直到收敛。
        2. 每一次滑动到新的区域，计算滑动窗口内的均值来作为中心点，滑动窗口内的点的数量为窗口内的密度。在每一次移动中，窗口会想密度更高的区域移动。
        3. 移动窗口，计算窗口内的中心点以及窗口内的密度，知道没有方向在窗口内可以容纳更多的点，即一直移动到圆内密度不再增加为止。
        4. 步骤一到三会产生很多个滑动窗口，当多个滑动窗口重叠时，保留包含最多点的窗口，然后根据数据点所在的滑动窗口进行聚类。
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.cluster import MeanShift
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

# 创建画布
fig = plt.figure(figsize=(3, 13))
plt.subplots_adjust(
    left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01
)


# 设置数据集
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

varied = datasets.make_blobs(
    n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170
)

# 设置模型
datasets = [noisy_circles, noisy_moons, blobs, no_structure, aniso, varied]
for i, dataset in enumerate(datasets):
    # 标准化
    x = StandardScaler().fit_transform(dataset[0])
    # 带宽
    bandwidth = cluster.estimate_bandwidth(x,quantile=0.3)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    y_pred = ms.fit_predict(dataset[0])
    ax = fig.add_subplot(6, 1, i + 1)
    plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=y_pred)
plt.show()
