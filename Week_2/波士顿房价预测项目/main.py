import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

from linear_model import Model
from linear_model import LinearRegression
from linear_model import GDRegressor
from linear_model import mean_squared_error

# 获取数据集
boston = load_boston()
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = boston.data
target = boston.target
boston_df = pd.DataFrame(np.hstack([data, target.reshape(-1, 1)]), columns=columns)

#数据分析
print(boston_df.head())
data_describe = boston_df.describe()
#数据可视化
index = data_describe.index[1:]
colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']
plt.figure(figsize=(12, 8))
bartarget = ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV']
for i in range(len(bartarget)):
    ax1 = plt.subplot(4, 3, i+1)
    ax1.set_title(bartarget[i])
    for j in range(len(index)):
        plt.bar(index[j], data_describe.loc[index[j], bartarget[i]], color=colors[j])
plt.subplots_adjust(wspace=0.2, hspace=0.7)
plt.show()

plt.figure(figsize=(12, 8))
chasnum = (len(boston_df[boston_df['CHAS'] == 1]), len(boston_df[boston_df['CHAS'] == 0]))
chas = ["By the river", "Not by the river"]
plt.pie(chasnum, labels=chas, autopct='%1.2f%%', explode=[0.1, 0], colors=['r', 'c'])
plt.title('Is it by the river?')
plt.show()

plt.figure(figsize=(12, 8))
for i in range(len(bartarget)):
    ax = plt.subplot(4, 4, i+1)
    ax.set_title(bartarget[i])
    plt.hist(boston_df.loc[:, bartarget[i]], color='green', bins=50)
plt.subplots_adjust(wspace=0.2, hspace=0.7)
plt.show()

plt.figure(figsize=(12, 8))
for i in range(len(bartarget[:-1])):
    ax = plt.subplot(4, 4, i+1)
    plt.scatter(boston_df.loc[:, bartarget[i]], boston_df['MEDV'].values, s=5, c='b', marker="o")
    plt.xlabel(bartarget[i])
    plt.ylabel('MEDV')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
Pear = np.array([])
for i in range(len(bartarget[:-1])):
    Pear = np.append(Pear, [pearsonr(boston_df.loc[:, bartarget[i]], boston_df['MEDV'].values)[0]])
plt.barh(bartarget[:-1], Pear, color='purple')
plt.grid()
plt.show()


#数据处理
#低方差特征过滤
transfer = VarianceThreshold()
data_new = transfer.fit_transform(data)

#主成分分析
transfer = PCA()
data_new = transfer.fit_transform(data_new)

#划分数据集
x_train, x_test, y_train, y_test = train_test_split(data_new, target, random_state=29)

#特征工程：无量纲化-标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#预估器
w1 = LinearRegression(x_train, y_train)
w2 = GDRegressor(x_train, y_train, 0.01, 1000)


#加载模型
#w1 = np.load("LinearRegression.npy")
#w2 = np.load("GDRegressor.npy")

#得出模型
print("正规方程偏重:\n", w1[0])
print("正规方程权重系数:\n", w1[1:])
print("梯度下降偏重:\n", w2[0])
print("梯度下降权重系数:\n", w2[1:])

#模型评估
y_predict_1 = Model(x_test, w1)
print("正规方程预测房价:\n", y_predict_1)
error = mean_squared_error(y_test, y_predict_1)
print("正规方程均方误差:\n", error)
y_predict_2 = Model(x_test, w2)
print("梯度下降预测房价:\n", y_predict_2)
error = mean_squared_error(y_test, y_predict_2)
print("梯度下降均方误差:\n", error)


#模型可视化
plt.figure(figsize=(12, 8))
ax1 = plt.subplot(2, 1, 1)
plt.title("正规方程——训练集", fontproperties="SimHei", fontsize=20)
plt.xlabel("index")
plt.ylabel("MEDV")
plt.plot(np.arange(0, x_train.shape[0]), y_train, 'r--')
plt.plot(np.arange(0, x_train.shape[0]), Model(x_train, w1), 'g-.')
ax2 = plt.subplot(2, 1, 2)
plt.title("正规方程——测试集", fontproperties="SimHei", fontsize=20)
plt.xlabel("index")
plt.ylabel("MEDV")
plt.plot(np.arange(0, x_test.shape[0]), y_test, 'r--')
plt.plot(np.arange(0, x_test.shape[0]), y_predict_1, 'g-.')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
ax3 = plt.subplot(2, 1, 1)
plt.title("梯度下降——训练集", fontproperties="SimHei", fontsize=20)
plt.xlabel("index")
plt.ylabel("MEDV")
plt.plot(np.arange(0, x_train.shape[0]), y_train, 'r--')
plt.plot(np.arange(0, x_train.shape[0]), Model(x_train, w2), 'g-.')
ax4 = plt.subplot(2, 1, 2)
plt.title("梯度下降——测试集", fontproperties="SimHei", fontsize=20)
plt.xlabel("index")
plt.ylabel("MEDV")
plt.plot(np.arange(0, x_test.shape[0]), y_test, 'r--')
plt.plot(np.arange(0, x_test.shape[0]), y_predict_2, 'g-.')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
ax5 = plt.subplot(2, 1, 1)
plt.title("正规方程(红)与梯度下降(绿)——训练集对比", fontproperties="SimHei", fontsize=20)
plt.xlabel("index")
plt.ylabel("MEDV")
plt.plot(np.arange(0, x_train.shape[0]), Model(x_train, w1), 'r--')
plt.plot(np.arange(0, x_train.shape[0]), Model(x_train, w2), 'g-.')

ax6 = plt.subplot(2, 1, 2)
plt.title("正规方程(红)与梯度下降(绿)——测试集对比", fontproperties="SimHei", fontsize=20)
plt.xlabel("index")
plt.ylabel("MEDV")
plt.plot(np.arange(0, x_test.shape[0]), y_predict_2, 'g--')
plt.plot(np.arange(0, x_test.shape[0]), y_predict_1, 'r-.')
plt.tight_layout()
plt.show()

#保存模型
np.save("LinearRegression.npy", w1)
np.save("GDRegressor.npy", w2)
