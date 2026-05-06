import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# 查看数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# 数据预处理
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.


def sigmoid(Z):
    """
    Sigmoid激活函数
    """
    A = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # 防止指数溢出
    cache = Z
    return A, cache


def relu(Z):
    """
    ReLU激活函数
    """
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    """
    ReLU函数的反向传播
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, cache):
    """
    Sigmoid函数的反向传播
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def initialize_parameters_deep(layer_dims):
    """
    初始化深度神经网络的参数
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # 网络层数

    for l in range(1, L):
        # 使用He初始化方法
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2.0 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """
    线性前向传播: Z = W*A + b
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    线性+激活前向传播
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
    """
    多层网络前向传播
    """
    caches = []
    A = X
    L = len(parameters) // 2  # 网络层数

    # 前L-1层使用ReLU激活函数
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters["W" + str(l)],
                                             parameters["b" + str(l)],
                                             activation="relu")
        caches.append(cache)

    # 最后一层使用sigmoid激活函数
    AL, cache = linear_activation_forward(A,
                                          parameters["W" + str(L)],
                                          parameters["b" + str(L)],
                                          activation="sigmoid")
    caches.append(cache)

    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


def compute_cost(AL, Y, parameters, lambd=0.01):
    """
    计算成本函数（包含L2正则化）
    """
    m = Y.shape[1]

    # 交叉熵成本
    cost = -np.sum(Y * np.log(AL + 1e-8) + (1 - Y) * np.log(1 - AL + 1e-8)) / m

    # L2正则化项
    L = len(parameters) // 2
    l2_regularization_cost = 0
    for l in range(1, L + 1):
        l2_regularization_cost += np.sum(np.square(parameters["W" + str(l)]))
    l2_regularization_cost *= lambd / (2 * m)

    cost = cost + l2_regularization_cost
    return cost


def linear_backward(dZ, cache, lambd=0.01):
    """
    线性层的反向传播
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m + (lambd / m) * W
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambd=0.01):
    """
    线性+激活层的反向传播
    """
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, lambd=0.01):
    """
    多层网络反向传播
    """
    gradients = {}
    L = len(caches)  # 网络层数
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # 初始化输出层梯度
    dAL = -(np.divide(Y, AL + 1e-8) - np.divide(1 - Y, 1 - AL + 1e-8))

    # 输出层（sigmoid）反向传播
    current_cache = caches[L - 1]
    gradients["dA" + str(L - 1)], gradients["dW" + str(L)], gradients["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, activation="sigmoid", lambd=lambd)

    # 隐藏层（relu）反向传播
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(gradients["dA" + str(l + 1)], current_cache, activation="relu", lambd=lambd)
        gradients["dA" + str(l)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp

    return gradients


def update_parameters(parameters, gradients, learning_rate):
    """
    更新参数
    """
    L = len(parameters) // 2  # 网络层数

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * gradients["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * gradients["db" + str(l)]

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, lambd=0.01):
    """
    深度神经网络模型
    """
    np.random.seed(1)
    costs = []

    # 初始化参数
    parameters = initialize_parameters_deep(layers_dims)

    # 梯度下降循环
    for i in range(0, num_iterations):

        # 前向传播
        AL, caches = L_model_forward(X, parameters)

        # 计算成本
        cost = compute_cost(AL, Y, parameters, lambd)

        # 反向传播
        gradients = L_model_backward(AL, Y, caches, lambd)

        # 更新参数
        parameters = update_parameters(parameters, gradients, learning_rate)

        # 打印成本
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    return parameters, costs


def predict_with_model(X, parameters):
    """
    使用训练好的模型进行预测
    """
    m = X.shape[1]
    n = len(parameters) // 2  # 网络层数
    p = np.zeros((1, m))

    # 前向传播
    probas, caches = L_model_forward(X, parameters)

    # 将概率转换为预测结果
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


if __name__ == '__main__':
    # 定义网络结构 [输入层, 隐藏层1, 隐藏层2, 输出层]
    layers_dims = [train_set_x.shape[0], 20, 7, 5, 1]  # 4层网络

    # 训练深度神经网络
    parameters, costs = L_layer_model(train_set_x, train_set_y, layers_dims,
                                      num_iterations=2500, learning_rate=0.0075,
                                      print_cost=True, lambd=0.01)

    # 训练集预测
    train_predictions = predict_with_model(train_set_x, parameters)
    train_accuracy = 100 - np.mean(np.abs(train_predictions - train_set_y)) * 100
    print("Train Accuracy: {} %".format(train_accuracy))

    # 测试集预测
    test_predictions = predict_with_model(test_set_x, parameters)
    test_accuracy = 100 - np.mean(np.abs(test_predictions - test_set_y)) * 100
    print("Test Accuracy: {} %".format(test_accuracy))