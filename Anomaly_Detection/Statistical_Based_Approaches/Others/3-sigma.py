"""
1、参考文章：
    https://zhuanlan.zhihu.com/p/36297816
2、原理：
    若数据服从正态分布，则异常值被定义为一组结果值中与平均值的偏差超过三倍标准差 σ 的值。
    若数据不服从正态分布，也可以用远离平均值的多少倍标准差来描述
"""

import pandas as pd
import matplotlib.pyplot as plt

# 处理数据
sale_data = pd.read_excel('catering_sale.xls')
sale_data.drop('利润', axis=1, inplace=True)
sale_data.columns = ['date', 'sale_amount']

data_x = sale_data['date']
data_y = sale_data['sale_amount']
# 获取均值和标准差，设定阈值
sale_mean = sale_data['sale_amount'].describe()[1]
sale_std = sale_data['sale_amount'].describe()[2]
threshold1 = sale_mean - 3 * sale_std
threshold2 = sale_mean + 3 * sale_std

# 保存异常值
outlier_x = []
outlier_y = []
for i in range(len(data_y)):
    if data_y[i] > threshold2 or data_y[i] < threshold1:
        outlier_y.append(data_y[i])
        outlier_x.append(data_x[i])

# 绘图
plt.plot(data_x, data_y)
plt.scatter(outlier_x, outlier_y, color='red')
plt.show()
