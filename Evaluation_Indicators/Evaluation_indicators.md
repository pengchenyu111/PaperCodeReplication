# 机器学习模型评估指标汇总 

参考文章：

1. https://www.cnblogs.com/zongfa/p/9431807.html
2. https://blog.csdn.net/liuy9803/article/details/80762862
3. https://www.jianshu.com/p/9ee85fdad150

Sklearn上的api：

​	https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

------



## 1、分类问题

```
真正(True Positive , TP)：被模型预测为正的正样本。
假正(False Positive , FP)：被模型预测为正的负样本。
假负(False Negative , FN)：被模型预测为负的正样本。
真负(True Negative , TN)：被模型预测为负的负样本。

真正率(True Positive Rate,TPR)：TPR=TP/(TP+FN)，即被预测为正的正样本数 /正样本实际数。
假正率(False Positive Rate,FPR) ：FPR=FP/(FP+TN)，即被预测为正的负样本数 /负样本实际数。
假负率(False Negative Rate,FNR) ：FNR=FN/(TP+FN)，即被预测为负的正样本数 /正样本实际数。
真负率(True Negative Rate,TNR)：TNR=TN/(TN+FP)，即被预测为负的负样本数 /负样本实际数/2
```

- 混淆矩阵

- 准确率

  正确预测的正反例数 /总数

  Accuracy = (TP+TN)/(TP+FN+FP+TN)

- 精确率

  正确预测的正例数 /预测正例总数

  Precision = TP/(TP+FP)

- 召回率

  正确预测的正例数 /实际正例总数

  Recall = TP/(TP+FN)

- F1 Score

  2/F1 = 1/Precision + 1/Recall

- ROC曲线

  横坐标为False Positive Rate(FPR假正率)，纵坐标为True Positive Rate(TPR真正率)

  越靠近左上越好

- AUC曲线

  AUC值(面积)越大的分类器，性能越好

- PR曲线

  PR曲线的横坐标是精确率P，纵坐标是召回率R

------

## 2、回归问题

- MAE：平均绝对误差

- MSE：均方误差

- RMSE：均方根误差

- R Squared：R方误差

  结果是0：约等于瞎猜。
  结果是1：模型无错误。
  结果是0-1：就是我们模型的好坏程度。
  结果是负数：不如瞎猜（其实导致这种情况说明我们的数据其实没有啥线性关系）

------

## 3、聚类问题

### 3.1 外部评价指标（有标签）

- v-measure

  **均一性**（Homogeneity）指每个簇中只包含单个类别的样本。如果一个簇中的类别只有一个，则均一性为1；如果有多个类别，计算该类别下的簇的条件经验熵H(C|K)，值越大则均一性越小。

  **完整性**（Completeness）指同类别样本被归类到相同的簇中。如果同类样本全部被分在同一个簇中，则完整性为1；如果同类样本被分到不同簇中，计算条件经验熵H(K|C)，值越大则完整性越小。

  单独考虑均一性或完整性都是片面的，因此引入两个指标的加权平均V-measure。如果β>1则更注重完整性，如果β<1则更注重均一性。

- 兰德指数RI、调整兰德指数ARI

  计算样本预测值与真实值之间的相似度，RI取值范围是[0,1]，值越大意味着聚类结果与真实情况越吻合。

  调整兰德指数：

  在聚类结果随机产生的情况下，指标应该接近零。 ARI取值范围为[−1,1]，值越大意味着聚类结果与真实情况越吻合。

- 互信息MI、标准互信息NMI

### 3.2 内部评价指标（无标签）

- Davies-Boulding指数（DBI）

  又称分类适确性指标，计算两个簇Ci、Cj各自的样本间平均距离avg(C)之和除以两个簇中心点μ之间的距离，DBI越小说明聚类效果越好。由于DBI使用欧氏距离，对环状分布的数据效果很差。

- Dunn 指数（DI）

- 轮廓系数（Silhouette Coefficient）

  轮廓系数的值介于[-1,1]，

  越接近于1表示样本i聚类越合理；

  越接近-1，表示样本i越应该被分类到其它簇中；

  越接近于0，表示样本应该在边界上。

  所有样本的si均值被称为聚类结果的轮廓系数。

