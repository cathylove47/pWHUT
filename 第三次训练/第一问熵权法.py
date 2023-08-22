import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置图片dpi
plt.rcParams['figure.dpi'] = 300

# 导入数据
"""
数据预处理
"""
df_food_raw = pd.read_excel('附件2 慢性病及相关因素流调数据.xlsx',sheet_name='饮食习惯')
# 不读取后面7列
df_food = df_food_raw.iloc[:,:-7]
df_food1 = df_food.fillna(-1).values.copy()
j = 0
for i in range(df_food1.shape[0]):
    for j in range(0,df_food1.shape[1],5):
        if df_food1[i,j]>0:
            pass
        elif np.any(df_food1[i,j+1:j+5] >0):
            df_food1[i,j] = 1
        else:
            df_food1[i,j] = 2
j = 0
for i in range(df_food1.shape[0]):
    for j in range(0,df_food1.shape[1],5):
        if np.all(df_food1[i,j+1:j+5]<0):
            df_food1[i,j] =2
j = 0
for i in range(df_food1.shape[0]):
    for j in range(0,df_food1.shape[1],5):
        if df_food1[i,j]==2: # 如果不吃
            df_food1[i,j+1] = 0 # 周为0
            df_food1[i,j+4] = 0 # 量为0
        else:
            if df_food1[i,j+1]>0:
                pass
            elif df_food1[i,j+2]>0:
                # 周为日的7倍
                df_food1[i,j+1] = (df_food1[i,j+2].copy())/7
            elif df_food1[i,j+3]>0:
                # 月频率为日的30倍
                df_food1[i,j+1] = (df_food1[i,j+1].copy())/30
for j in range(0,df_food1.shape[1],5):
    m = np.mean(df_food1[df_food1[:,j+1]>0,j+1])
    df_food1[df_food1[:,j+1]<0,j+1] = round(m,2)

for i in range(df_food1.shape[0]):
    for j in range(0,df_food1.shape[1],5):
        if df_food1[i,j+4]<0:
            df_food1[i,j+4] = round(np.mean(df_food1[df_food1[:,j+1]==df_food1[i,j+1],j+4]),2)
        if df_food1[i,j+4]<0:
            df_food1[i,j+4] = round(np.mean(df_food1[:,j+4]),2)

"""
去掉周,月
"""
df_food2 = df_food1[:,np.array([0,1,4])+0*5].copy()
for j in range(5,df_food1.shape[1],5):
    df_food2 = np.c_["1",df_food2,df_food1[:,np.array([0,1,4])+j].copy()]

df_food2_df = pd.DataFrame(df_food2,columns=[i for i in df_food.columns.values if i[0]!="U"])
"""
统计每一行中食用某种食物的频率不为0的个数,比如第一行中,选取第二列,第五列,第八列,以此类推
"""
# 先选取第二列,第五列,第八列,以此类推
df_food3 = df_food2_df.iloc[:,np.arange(1,df_food2_df.shape[1],3)].copy()
# 统计每一行中不为0的个数
df_food2_df["日摄入食用种类"] = df_food3[df_food3>0.3].count(axis=1)
print(df_food2_df["日摄入食用种类"].describe())
# 统计每个人每天吃新鲜蔬菜的量,等于吃蔬菜的频率乘以使用量乘以50
df_food2_df["每日新鲜蔬菜食用量"] = df_food2_df["食用新鲜蔬菜的频率"]*df_food2_df["平均每次食用量.18"]*50
print(df_food2_df["每日新鲜蔬菜食用量"].describe())
df_food2_df["每日水果食用量"] = df_food2_df["食用水果的频率"]*df_food2_df["平均每次食用量.24"]*50
print(df_food2_df["每日水果食用量"].describe())
df_food2_df["每日奶制品使用量"] = df_food2_df["食用鲜奶的频率"]*df_food2_df["平均每次食用量.10"]*50 + df_food2_df["食用酸奶的频率"]*df_food2_df["平均每次食用量.12"]*50
print(df_food2_df["每日奶制品使用量"].describe())
df_food2_df["每天摄入鱼禽、蛋类和瘦肉的量"] = (df_food2_df["食用水产的频率"]*df_food2_df["平均每次食用量.9"] \
+ df_food2_df["食用禽肉的频率"]*df_food2_df["平均每次食用量.7"] + df_food2_df["食用蛋类的频率"]*df_food2_df["平均每次食用量.13"] +\
df_food2_df["食用牛羊肉的频率"]*df_food2_df["平均每次食用量.6"]*0.6 + df_food2_df["食用猪肉的频率"] * df_food2_df["平均每次食用量.5"]*0.6)*50
print(df_food2_df["每天摄入鱼禽、蛋类和瘦肉的量"].describe())
df_food2_df["每天摄入油的量"] = df_food_raw["植物油"]
df_food2_df.loc[df_food2_df["每天摄入油的量"]==0,"每天摄入油的量"] = 7
print(df_food2_df["每天摄入油的量"].describe())
df_food2_df["每天摄入盐的量"] = df_food_raw["盐"]
# 如果为0就改为15
df_food2_df.loc[df_food2_df["每天摄入盐的量"]==0,"每天摄入盐的量"] = 15
print(df_food2_df["每天摄入盐的量"].describe())
"""
df_food2_df的最后七列作为评价指标
进行熵权法计算熵权值
"""
# 先选取df_food2_df的最后七列
Data = df_food2_df.iloc[:,-7:].copy()
data_raw = df_food2_df.iloc[:,-7:].copy()
# 进行标准化
# 对日摄入食用种类进行正向标准化
Data["日摄入食用种类"] = (Data["日摄入食用种类"]-Data["日摄入食用种类"].min())/(Data["日摄入食用种类"].max()-Data["日摄入食用种类"].min())
# 对每日新鲜蔬菜食用量进行正向标准化
Data["每日新鲜蔬菜食用量"] = (Data["每日新鲜蔬菜食用量"]-Data["每日新鲜蔬菜食用量"].min())/(Data["每日新鲜蔬菜食用量"].max()-Data["每日新鲜蔬菜食用量"].min())
# 对每日水果食用量进行正向标准化
Data["每日水果食用量"] = (Data["每日水果食用量"]-Data["每日水果食用量"].min())/(Data["每日水果食用量"].max()-Data["每日水果食用量"].min())
# 对每日奶制品使用量进行正向标准化
Data["每日奶制品使用量"] = (Data["每日奶制品使用量"]-Data["每日奶制品使用量"].min())/(Data["每日奶制品使用量"].max()-Data["每日奶制品使用量"].min())
# 对每天摄入鱼禽、蛋类和瘦肉的量进行正向标准化
Data["每天摄入鱼禽、蛋类和瘦肉的量"] = (Data["每天摄入鱼禽、蛋类和瘦肉的量"]-Data["每天摄入鱼禽、蛋类和瘦肉的量"].min())/(Data["每天摄入鱼禽、蛋类和瘦肉的量"].max()-Data["每天摄入鱼禽、蛋类和瘦肉的量"].min())
# 对每天摄入油的量进行负向标准化
Data["每天摄入油的量"] = (Data["每天摄入油的量"].max()-Data["每天摄入油的量"])/(Data["每天摄入油的量"].max()-Data["每天摄入油的量"].min())
# 对每天摄入盐的量进行负向标准化
Data["每天摄入盐的量"] = (Data["每天摄入盐的量"].max()-Data["每天摄入盐的量"])/(Data["每天摄入盐的量"].max()-Data["每天摄入盐的量"].min())
# print(Data.head())
# 计算熵值
Data = np.array(Data)
# 计算每个指标的熵值
k = 1/np.log(Data.shape[0])
yij = Data.sum(axis=0)
pij = Data/yij
Lij = pij*np.log(pij)
Ej = -k*(Lij.sum(axis=0))
# 计算每个指标的权重
wi = (1-Ej)/(1-Ej).sum()
print(wi)
print(data_raw)
data_raw.insert(0, '每日新鲜蔬菜使用量1', 0)
data_raw.insert(0, '每日水果食用量1', 0)
data_raw.insert(0, '每日奶制品使用量1', 0)
data_raw.insert(0, '每天摄入鱼禽、蛋类和瘦肉的量1', 0)
data_raw.insert(0, '每天摄入油的量1', 0)
data_raw.insert(0, '每天摄入盐的量1', 0)
data_raw.insert(0, '日摄入食用种类1', 0)
# 遍历每日新鲜蔬菜食用量,如果大于等于300克就为1,否则为0
data_raw.loc[data_raw["每日新鲜蔬菜食用量"]>=300,"每日新鲜蔬菜食用量1"] = 1
data_raw.loc[data_raw["每日新鲜蔬菜食用量"]<300,"每日新鲜蔬菜食用量1"] = 0
# 遍历每日水果食用量,如果大于等于200克就为1,否则为0
data_raw.loc[data_raw["每日水果食用量"]>=150,"每日水果食用量1"] = 1
data_raw.loc[data_raw["每日水果食用量"]<150,"每日水果食用量1"] = 0
# 遍历每日奶制品使用量,如果大于等于500克就为0,否则为1
data_raw.loc[data_raw["每日奶制品使用量"]>=250,"每日奶制品使用量1"] = 0
data_raw.loc[data_raw["每日奶制品使用量"]<250,"每日奶制品使用量1"] = 1
# 遍历每天摄入鱼禽、蛋类和瘦肉的量,如果大于等于200克就为1,否则为0
data_raw.loc[data_raw["每天摄入鱼禽、蛋类和瘦肉的量"]>=200,"每天摄入鱼禽、蛋类和瘦肉的量1"] = 1
data_raw.loc[data_raw["每天摄入鱼禽、蛋类和瘦肉的量"]<200,"每天摄入鱼禽、蛋类和瘦肉的量1"] = 0
# 遍历每天摄入油的量,如果大于等于9克就为0,否则为1
data_raw.loc[data_raw["每天摄入油的量"]>=9,"每天摄入油的量1"] = 0
data_raw.loc[data_raw["每天摄入油的量"]<9,"每天摄入油的量1"] = 1
# 遍历每天摄入盐的量,如果大于等于15克就为0,否则为1
data_raw.loc[data_raw["每天摄入盐的量"]>=15,"每天摄入盐的量1"] = 0
data_raw.loc[data_raw["每天摄入盐的量"]<15,"每天摄入盐的量1"] = 1
# 日摄入食用种类,如果大于等于12种就为1,否则为0
data_raw.loc[data_raw["日摄入食用种类"]>=12,"日摄入食用种类1"] = 1
data_raw.loc[data_raw["日摄入食用种类"]<12,"日摄入食用种类1"] = 0
print(data_raw)
# 计算得分
data_raw["得分"] = 0.25148*data_raw["每日新鲜蔬菜食用量1"]+0.20256*data_raw["每天摄入鱼禽、蛋类和瘦肉的量1"]+ \
                   0.15156*data_raw["每日奶制品使用量1"]+0.10123*data_raw["日摄入食用种类1"] + \
                   0.07562*data_raw["每天摄入油的量1"]+ 0.02520*data_raw["每天摄入盐的量1"] + 0.19325*data_raw["每日水果食用量1"]
print(data_raw["得分"].describe())
# 统计得分小于0.5的人数
print(data_raw.loc[data_raw["得分"]<0.6,"得分"].count())
# 统计蔬菜为0的个数
print(data_raw.loc[data_raw["每日新鲜蔬菜食用量1"]==0,"每日新鲜蔬菜食用量1"].count())
# 统计食物种类为0的个数
print(data_raw.loc[data_raw["日摄入食用种类1"]==0,"日摄入食用种类1"].count())
# 统计新鲜水果为0的个数
print(data_raw.loc[data_raw["每日水果食用量1"]==0,"每日水果食用量1"].count())
# 统计奶制品为0的个数
print(data_raw.loc[data_raw["每日奶制品使用量1"]==0,"每日奶制品使用量1"].count())
# 统计鱼禽蛋肉为0的个数
print(data_raw.loc[data_raw["每天摄入鱼禽、蛋类和瘦肉的量1"]==0,"每天摄入鱼禽、蛋类和瘦肉的量1"].count())
# 统计油为.0的个数
print(data_raw.loc[data_raw["每天摄入油的量1"]==0,"每天摄入油的量1"].count())
# 统计盐为0的个数
print(data_raw.loc[data_raw["每天摄入盐的量1"]==0,"每天摄入盐的量1"].count())
# 导入lightgbm模型
import lightgbm as lgb
# 导入新的列
data_new = pd.read_excel("附件2 慢性病及相关因素流调数据.xlsx",sheet_name="流行病学调查数据")
# 选取出生,性别,文化程度,婚姻状况,职业这几列
data_new = data_new[["出生年","性别","文化程度","婚姻状况","职业"]]
# 将出生年份转换为年龄
data_new["出生年"] = 2023 - data_new["出生年"]
# 将data_new和data_raw合并
data = pd.concat([data_raw,data_new],axis=1)
print(data)
# 将data保存为excel
data.to_excel("第一问结束的数据.xlsx",index=False)

