import numpy as np
import time
import matplotlib.pyplot as plt

#计时开始
tic = time.time()

# 输入数据：样本
x = np.array([-2, -1, 0, 1, 2])  # 样本x
y = 1/(1+np.exp(-x))  # 样本y：1/(1+np.exp(-x))
m = len(x)  # 样本容量

# 赋初值
alpha = 0.01  # 学习率
w = 1
b = 1
i = 0
n_max = 1000
L = np.empty(n_max)  # 损失函数
J = np.empty(n_max)  # 代价函数

# 第一层循环：采用梯度下降法计算w，b值
while (i<n_max):
    sumdL_w = 0  # 损失函数对w的导数赋初值
    sumdL_b = 0  # 损失函数对b的导数赋初值
    sumJ = 0  # 代价函数赋初值 

    # 第二层循环：遍历样本，计算损失函数、代价函数及其导数
    for j in range(m):
        # 计算假设函数及其导数
        z = w * x[j] + b
        yhat = 1 / (1 + np.exp(-z))  # 逻辑回归预测值
        dyhat_z = np.exp(-z) / ((1 + np.exp(-z)) ** 2)  # yhat对z的偏导数
        dz_w = x[j]  # z对w的偏导数，以下命名规则类似
        dz_b = 1
        dyhat_w = dyhat_z * dz_w
        dyhat_b = dyhat_z * dz_b

        # 计算损失函数及其导数
        L = -y[j]*np.log(yhat) - (1-y[j])*np.log(1-yhat)
        dL_yhat = -y[j] / yhat + (1 - y[j]) / (1 - yhat)
        dL_w = dL_yhat * dyhat_w
        dL_b = dL_yhat * dyhat_b

        # 计算代价函数及其导数
        sumJ = sumJ + L
        sumdL_w = sumdL_w + dL_w
        sumdL_b = sumdL_b + dL_b

    J[i] = sumJ/m
    dJ_w = sumdL_w / m
    dJ_b = sumdL_b / m

    # 梯度下降迭代计算w,b
    w = w - alpha * dJ_w
    b = b - alpha * dJ_b
    print('i=',i, 'w=',w, 'b=',b, 'J[i]=',J[i])
    i = i + 1

# 计时结束
toc = time.time()
tictoc = toc - tic
print('total time of CPU is', tictoc, 's')
plt.plot(J)
plt.show()