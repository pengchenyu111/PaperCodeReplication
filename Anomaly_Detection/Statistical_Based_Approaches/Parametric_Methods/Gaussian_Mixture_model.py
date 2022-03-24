"""
1、参考文章：
    https://blog.csdn.net/Zhang_Pro/article/details/105849582
    https://blog.csdn.net/zeronose/article/details/104737115
2、单一高斯模型GSM：
    模拟具有单一中心点的数据，拟合效果较好；对于多数据中心点时，拟合效果较差
3、混合高斯模型GMM：
    通过求解多个高斯模型，用一定的权重将几个高斯模型融合成为一个模型，即高斯混合模型
    EM算法：
        用似然估计来求确定高斯分布的 μ 和 σ
            似然估计：通过使得样本集的联合概率最大来对参数进行估计，从而选择最佳的分布模型（利用已知样本结果，反推最有可能（最大概率）导致这样结果的参数值）
        基本思想：E步：固定θ，优化Q；M步：固定Q，优化θ；交替将极值推向最大（即期望最大）。
            首先根据己经给出的观测数据，估计出模型参数的值；
            然后再依据上一步估计出的参数值估计缺失数据的值，再根据估计出的缺失数据加上之前己经观测到的数据重新再对参数值进行估计，
            然后反复迭代，直至最后收敛，迭代结束。
4、应用：
    聚类
5、贝叶斯高斯混合模型

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

# 不使用科学计数法
np.set_printoptions(suppress=True)

# 生成随机样本，2个簇
n_samples = 500
np.random.seed(0)
C = np.array([[0.2, -0.3], [1.7, 0.4]])
X = np.r_[
    np.dot(np.random.randn(n_samples, 2), C),
    0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
]

# 使用高斯混合模型
GMM = GaussianMixture(n_components=5, covariance_type='full')
GMM.fit(X)
# 预测的结果是属于哪个高斯模型
Y = GMM.predict(X)
# 绘制结果
color_list = ["navy", "c", "cornflowerblue", "gold", "darkorange"]
for i, color in enumerate(color_list):
    plt.scatter(X[Y == i, 0], X[Y == i, 1], 0.8, color=color)
plt.show()

# 使用贝叶斯高斯混合模型
BGMM = BayesianGaussianMixture(n_components=5, covariance_type='full')
BGMM.fit(X)
bgmm_Y = BGMM.predict(X)
# 绘制结果
color_list = ["navy", "c", "cornflowerblue", "gold", "darkorange"]
for i, color in enumerate(color_list):
    plt.scatter(X[bgmm_Y == i, 0], X[bgmm_Y == i, 1], 0.8, color=color)

plt.show()

