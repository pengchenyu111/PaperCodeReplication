"""
    OPTICS算法（Ordering Points to identify the clustering structure）

    1、参考文章
        https://zhuanlan.zhihu.com/p/77052675
        https://zhuanlan.zhihu.com/p/41930932
    2、原理：
        （是对DBSCAN的一种改进方法，用于解决DBSCAN对r和minPoints参数敏感的问题）
        该算法中并不显式的生成数据聚类，只是对数据集合中的对象进行排序，得到一个有序的对象列表，通过该有序列表，可以得到一个决策图
    3、
        需要调参的参数变成了3个

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import OPTICS
from itertools import cycle, islice
from sklearn.preprocessing import StandardScaler

# 创建画布
fig = plt.figure(figsize=(3, 13))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01)
colors_picker = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00", ]

# 创建数据集
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)

# 设置模型
x = StandardScaler().fit_transform(noisy_circles[0])
pre1 = OPTICS(min_samples=20, xi=0.25, min_cluster_size=0.1).fit_predict(x)
ax = fig.add_subplot(6, 1, 1)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre1) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(noisy_circles[0][:, 0], noisy_circles[0][:, 1], color=colors[pre1])

x = StandardScaler().fit_transform(noisy_moons[0])
pre2 = OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.1).fit_predict(x)
ax = fig.add_subplot(6, 1, 2)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre2) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(noisy_moons[0][:, 0], noisy_moons[0][:, 1], color=colors[pre2])

x = StandardScaler().fit_transform(varied[0])
pre3 = OPTICS(min_samples=5, xi=0.035, min_cluster_size=0.2).fit_predict(x)
ax = fig.add_subplot(6, 1, 3)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre3) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(varied[0][:, 0], varied[0][:, 1], color=colors[pre3])

from collections import Counter

x = StandardScaler().fit_transform(aniso[0])
pre4 = OPTICS(min_samples=20, xi=0.1, min_cluster_size=0.2).fit_predict(x)
ax = fig.add_subplot(6, 1, 4)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre4) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(aniso[0][:, 0], aniso[0][:, 1], color=colors[pre4])

x = StandardScaler().fit_transform(blobs[0])
pre5 = OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.1).fit_predict(x)
ax = fig.add_subplot(6, 1, 5)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre5) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(blobs[0][:, 0], blobs[0][:, 1], color=colors[pre5])

x = StandardScaler().fit_transform(no_structure[0])
pre6 = OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.1).fit_predict(x)
ax = fig.add_subplot(6, 1, 6)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre6) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(no_structure[0][:, 0], no_structure[0][:, 1], color=colors[pre6])

plt.show()
