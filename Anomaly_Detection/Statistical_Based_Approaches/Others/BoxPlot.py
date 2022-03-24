"""
1、参考文章：
    https://blog.csdn.net/weixin_44322234/article/details/118142360
    https://zhuanlan.zhihu.com/p/148306737
2、原理:

"""

import pandas as pd
import matplotlib.pyplot as plt

# 处理数据
sale_data = pd.read_excel('catering_sale.xls')
sale_data.drop('利润', axis=1, inplace=True)
sale_data.columns = ['date', 'sale_amount']

plt.boxplot(x=sale_data['sale_amount'],
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
            meanprops={'marker': 'D', 'markerfacecolor': 'indianred'})
plt.show()
