"""
1、参考文章：
    https://blog.csdn.net/hansome_hong/article/details/107596543
    https://www.biaodianfu.com/dbscan.html
2、原理：

3、疑问
    DBSCAN对eps这个参数很敏感，而且还是无监督方法，那么怎么确定最好的eps呢？
    ====>聚类方法的评价指标：
                轮廓系数，越大越好  https://blog.csdn.net/qq_41684240/article/details/108175984

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from itertools import cycle, islice

# 创建画布
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

fig = plt.figure(figsize=(3, 13))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01)
colors_picker = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00", ]

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

varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)

# 设置模型
x = StandardScaler().fit_transform(noisy_circles[0])
pre1 = DBSCAN(eps=0.3).fit_predict(x)
ax = fig.add_subplot(6, 1, 1)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre1) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(noisy_circles[0][:, 0], noisy_circles[0][:, 1], color=colors[pre1])

x = StandardScaler().fit_transform(noisy_moons[0])
pre2 = DBSCAN(eps=0.3).fit_predict(x)
ax = fig.add_subplot(6, 1, 2)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre2) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(noisy_moons[0][:, 0], noisy_moons[0][:, 1], color=colors[pre2])

x = StandardScaler().fit_transform(varied[0])
pre3 = DBSCAN(eps=0.18).fit_predict(x)
ax = fig.add_subplot(6, 1, 3)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre3) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(varied[0][:, 0], varied[0][:, 1], color=colors[pre3])

from collections import Counter

x = StandardScaler().fit_transform(aniso[0])
pre4 = DBSCAN(eps=0.15).fit_predict(x)
ax = fig.add_subplot(6, 1, 4)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre4) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(aniso[0][:, 0], aniso[0][:, 1], color=colors[pre4])

x = StandardScaler().fit_transform(blobs[0])
pre5 = DBSCAN(eps=0.3).fit_predict(x)
ax = fig.add_subplot(6, 1, 5)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre5) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(blobs[0][:, 0], blobs[0][:, 1], color=colors[pre5])

x = StandardScaler().fit_transform(no_structure[0])
pre6 = DBSCAN(eps=0.3).fit_predict(x)
ax = fig.add_subplot(6, 1, 6)
colors = np.array(list(islice(cycle(colors_picker), int(max(pre6) + 1), )))
colors = np.append(colors, ["#000000"])
plt.scatter(no_structure[0][:, 0], no_structure[0][:, 1], color=colors[pre6])

plt.show()

# 使用轮廓系数来对聚类方法调参
best_score = 0
best_eps = 0
x = StandardScaler().fit_transform(varied[0])
for i in np.arange(0.15, 0.20, 0.01):
    dbscan = DBSCAN(eps=i).fit(x)
    labels = dbscan.labels_
    k = silhouette_score(x, labels)
    if k > best_score:
        best_score = k
        best_eps = i
    print("eps={} ====> 轮廓系数={}".format(i, k))
print("best_eps={} ====> 轮廓系数={}".format(best_eps, best_score))
