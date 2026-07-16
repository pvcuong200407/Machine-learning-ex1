import numpy as np
def computeCost(x, y, theta):
    m = x.shape[0]
    hypotheis = np.dot(x, theta)
    cost = np.dot(np.transpose(hypotheis - y),(hypotheis - y))
    cost /= 2*m
    return cost
