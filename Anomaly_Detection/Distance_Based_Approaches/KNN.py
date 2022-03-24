"""
1、参考文章：
    https://www.cnblogs.com/listenfwind/p/10311496.html
2、原理及思想:
    KNN的原理就是当预测一个新的值x的时候，根据它距离最近的K个点是什么类别来判断x属于哪个类别
    K 的取值和 距离 的计算方式至关重要
3、KNN和KMeans的联系:
    同：
        1、K值都是重点；2、都需要计算平面中点的距离
    异：
        KNN做分类；KMeans做聚类
4、关键：
    KNN是有监督算法，需要标签
"""
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 导入鸢尾花数据集
iris_data = load_iris()
train_x = iris_data.data
train_y = iris_data.target

# 建立模型
# 由于对于KNN来说，K的选择至关重要，所以我们一般会用GridSearch或交叉验证来计算较好的K
k_score = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(estimator=knn, X=train_x, y=train_y, cv=10, scoring='accuracy')
    k_score.append(cv_scores.mean())
best_k = k_score.index(max(k_score)) + 1
print("best k is {}".format())
plt.plot(range(1, 31), k_score)
plt.show()


