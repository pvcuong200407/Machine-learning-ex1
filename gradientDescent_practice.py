import numpy as np
from computeCostPractice import computeCost
def gradientDescent_1 (x, y, theta_init, alpha, iters_num):
    loss = [0]*97
    theta = theta_init
    j_history = np.zeros(iters_num)
    m = y.shape[0]
    for i in range (0, iters_num):
        j_history[i] = computeCost(x, y, theta)
        hyp = np.dot(x, theta)
        loss = hyp - y
        theta = theta - alpha*(np.dot(x.T, loss))*(1/m)

    return j_history, theta
def gradientDescent_2 (x, y, theta_init, alpha, iters_num):
    loss = [0]*97
    theta = np.array(theta_init, float)
    j_history = np.zeros(iters_num)
    m = y.shape[0]
    for i in range (0, iters_num):
        j_history[i] = computeCost(x, y, theta)
        hyp = np.dot(x,theta)
        loss = hyp - y
        theta = theta - alpha*(np.dot(x.T,loss))*(1/m)

    return j_history, theta