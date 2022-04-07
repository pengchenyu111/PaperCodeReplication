"""
    孤立森林
    1、参考文章：
        https://zhuanlan.zhihu.com/p/27777266
        https://blog.csdn.net/extremebingo/article/details/80108247
    2、原理：
        异常的前提：
            异常数据跟样本中大多数数据不太一样。
            异常数据在整体数据样本中占比比较小。
        IForest构建过程：
            1，从训练数据中随机选择 n 个点样本作为subsample，放入树的根节点。
            2，随机指定一个维度（attribute），在当前节点数据中随机产生一个切割点p——切割点产生于当前节点数据中指定维度的最大值和最小值之间。
            3，以此切割点生成了一个超平面，然后将当前节点数据空间划分为2个子空间：把指定维度里面小于p的数据放在当前节点的左孩子，把大于等于p的数据放在当前节点的右孩子。
            4，在孩子节点中递归步骤2和3，不断构造新的孩子节点，知道孩子节点中只有一个数据（无法再继续切割）或者孩子节点已达限定高度。
        预测过程：
            对于一个训练数据X，我们令其遍历每一颗iTree，然后计算X 最终落在每个树第几层（X在树的高度），最终我们可以得到X在每棵树的高度平均值
    3、评价：
        优点：
            1、时间复杂度较低；
            2、适用于大数据量
        缺点：
            1、不适用于高维数据：
                由于每次切数据空间都是随机选取一个维度，建完树后仍然有大量的维度信息没有被使用，导致算法可靠性降低；
                高维空间还可能存在大量噪音维度或者无关维度（irrelevant  attributes），影响树的构建；
            2、仅对全局稀疏点敏感，对局部稀疏点不敏感
"""

import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from Dataset.toy_dataset.anomaly_toy_dataset import data1, data2, data3, data4, data5

# 创建画布
fig = plt.figure(figsize=(3, 13))

x = StandardScaler().fit_transform(data1)
pre1 = IsolationForest(contamination=0.15, random_state=170).fit_predict(X=x)
fig.add_subplot(5, 1, 1)
plt.scatter(data1[:, 0], data1[:, 1], c=pre1)
print(pre1)

x = StandardScaler().fit_transform(data2)
pre2 = IsolationForest(contamination=0.15, random_state=170).fit_predict(X=x)
fig.add_subplot(5, 1, 2)
plt.scatter(data2[:, 0], data2[:, 1], c=pre2)
print(pre2)

x = StandardScaler().fit_transform(data3)
pre3 = IsolationForest(contamination=0.15, random_state=170).fit_predict(X=x)
fig.add_subplot(5, 1, 3)
plt.scatter(data3[:, 0], data3[:, 1], c=pre3)
print(pre3)

x = StandardScaler().fit_transform(data4)
pre4 = IsolationForest(contamination=0.15, random_state=170).fit_predict(X=x)
fig.add_subplot(5, 1, 4)
plt.scatter(data4[:, 0], data4[:, 1], c=pre4)
print(pre4)

x = StandardScaler().fit_transform(data5)
pre5 = IsolationForest(contamination=0.15, random_state=170).fit_predict(X=x)
fig.add_subplot(5, 1, 5)
plt.scatter(data5[:, 0], data5[:, 1], c=pre5)
print(pre5)

plt.show()
