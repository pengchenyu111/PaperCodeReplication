"""
1、参考文章：
    https://zhuanlan.zhihu.com/p/134923402
    https://blog.csdn.net/weixin_42056745/article/details/101287231
2、原理：
    (1) 首先我们选择一些类/组，并随机初始化它们各自的中心点。中心点是与每个数据点向量长度相同的位置。这需要我们提前预知类的数量(即中心点的数量)。
    (2) 计算每个数据点到中心点的距离，数据点距离哪个中心点最近就划分到哪一类中。
    (3) 计算每一类中中心点作为新的中心点。
    (4) 重复以上步骤，直到每一类中心在每次迭代后变化不大为止。也可以多次随机初始化中心点，然后选择运行结果最好的一个。
3、优化：
    1、K-Means++
        K-Means的缺点在于如果第一步中中心点的选择不好，则会产生大量地迭代步骤；
        K-Means++使用了一定的方法使得算法中的第一步（初始化中心）变得比较合理，而不是随机的选择中心
    2、MiniBatch
        数据量过大时，K-means收敛速度会很慢
        MiniBatch：随机从整体当中做一个抽样，选取出一小部分数据来代替整体
"""
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs

fig = plt.figure(figsize=(8, 3))

# 设置初始数据集
X_varied, y_varied = make_blobs(
    n_samples=10000, cluster_std=[1.0, 2.5, 0.5], random_state=170
)

# 设置模型
km = KMeans(n_clusters=3)
t0 = time.time()
y_pred = km.fit_predict(X_varied)
km_cost = time.time() - t0

# 显示中心点
ax = fig.add_subplot(1, 2, 1)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
km_centers = km.cluster_centers_
plt.scatter(km_centers[:, 0], km_centers[:, 1], c='red', marker='x')
ax.set_title("K-means cost: {:.4f}".format(km_cost))

# 设置MiniBatch模型
t1 = time.time()
mbkm = MiniBatchKMeans(n_clusters=3, batch_size=1024)
mbkm.fit_predict(X_varied)
mbkm_cost = time.time() - t1

# 显示中心点
ax = fig.add_subplot(1, 2, 2)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
mbkm_centers = mbkm.cluster_centers_
plt.scatter(mbkm_centers[:, 0], mbkm_centers[:, 1], c='red', marker='x')
ax.set_title("MiniBatch K-means cost: {:.4f}".format(mbkm_cost))

plt.show()
