import numpy as np

def mean_squared_error(y_train, y_predict):
    error = np.sum((y_predict - y_train) ** 2) / len(y_train)
    return error

def Model(x,w):
    x = np.insert(x, 0, 1, axis=1)
    y = np.dot(x, w.T)
    return y

def LinearRegression(x_train, y_train):
    x_train = np.insert(x_train, 0, 1, axis=1)
    w = np.linalg.inv(x_train.T.dot(x_train)).dot(x_train.T).dot(y_train)
    return w

def GDRegressor(x_train, y_train, eta, times):
    x_train = np.insert(x_train, 0, 1, axis=1)
    # 初始化权重系数
    n = x_train.shape[1]
    w = np.zeros(n)
    # 迭代更新权重系数
    prev_mse = float('inf')
    for i in range(times):
        # 计算预测值和误差
        y_predict = np.dot(x_train, w)
        error = y_train - y_predict
        # 计算梯度和更新权重系数
        grad = np.dot(x_train.T, error) / len(x_train)
        w += eta * grad
        # 计算均方误差并判断是否收敛
        mse = np.mean((y_train - np.dot(x_train, w)) ** 2)
        if abs(mse - prev_mse) <= 1e-5:
            break
        prev_mse = mse

    return w
