import pandas as pd
import numpy as np
from scipy.stats import entropy

# 定义评价指标和候选供应商数据
indicators = ['Price', 'Delivery Time', 'Quality']
suppliers = ['Supplier A', 'Supplier B', 'Supplier C']

# 定义每个指标对应的数据，这里使用了随机数据
data = np.array([[5, 20, 80], [6, 15, 90], [4, 18, 85]])
n = data.shape[0] # 对象个数
# 计算每个指标的信息熵
entropy_list = []
for i in range(len(indicators)):
    entropy_list.append(1/np.log(n)*entropy(data[:,i]))

# 计算每个指标的信息熵权重
weight_list = []
for entropy_i in entropy_list:
    weight_list.append(1- entropy_i)
weight_list = weight_list/sum(weight_list)
# 计算每个供应商的综合得分
scores = []
for i in range(len(suppliers)):
    score_i = 0
    for j in range(len(indicators)):
        score_i += weight_list[j] * data[i,j]
    scores.append(score_i)

# 找到得分最高的供应商
best_supplier = suppliers[np.argmax(scores)]

# 打印结果
print("各个指标的信息熵权重为：", weight_list)
print("每个供应商的综合得分为：", scores)
print("最佳供应商是：", best_supplier)
#