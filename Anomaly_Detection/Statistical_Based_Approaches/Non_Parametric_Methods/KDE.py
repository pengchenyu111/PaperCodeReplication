"""
1、参考文章：
    https://blog.csdn.net/unixtch/article/details/78556499
    https://blog.csdn.net/pipisorry/article/details/53635895
    https://zhuanlan.zhihu.com/p/150605951
2、核密度估计（Kernel Density Estimation，KDE）
    给定一个样本集，得到该样本集的分布密度函数；
3、缺点：
    计算代价大
    高维是难用

"""

import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# 生成随机数据集
rng = np.random.RandomState(42)
X = rng.random_sample((100, 2))

plt.scatter(X[:, 0], X[:, 1])
plt.show()

# 做KDE之前一般要做搜索，得出较好的 “带宽”
params = {"bandwidth": np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(X)
kde_best_bandwidth = grid.best_estimator_.bandwidth
print("best bandwidth: {0}".format(kde_best_bandwidth))
# 设置模型
kde = KernelDensity(kernel='gaussian', bandwidth=kde_best_bandwidth)
kde.fit(X)

print(kde.score_samples(X))
