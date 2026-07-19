import numpy as np
from computeCostPractice import computeCost
from gradientDescent_practice import gradientDescent_1
from gradientDescent_practice import gradientDescent_2
import matplotlib.pyplot as plt

xData = []
yData = []
with open('data/ex1data1.txt', 'r') as file:
    for line in file:
        if line.strip():
            col1, col2 = line.strip().split(',')
            xData.append(float(col1))
            yData.append(float(col2))
m = len(xData)
x = np.array([[1, xData[i]] for i in range(m)])
theta_init_1 = [0,0]
theta_init_2 = [0,0,0]
y = np.array(yData)
iter_num = 1500 #có thể thay đổi
alpha = 0.01 #có thể thay đổi
#Thử cost
cost = computeCost(x, y, theta_init_1)
print (f'Cost tìm được là: {cost:.2f}')
#Thử tìm theta,j trong hàm linear
j_1, theta_1 = gradientDescent_1(x, y, theta_init_1, alpha, iter_num)
print ('Theta_1 tìm được là: ', np.round(theta_1,2))
for i in range (0,5):
    print(f'j_1 tìm được: {j_1[i]:.5f}')
#Thử tìm theta, j trong hàm bậc 2
x_square = x[:,1]**2
xdata = x[:,1]
x_scaled = (xdata - np.mean(xdata)) / np.std(xdata)
x_square_scaled = (x_square - np.mean(x_square)) / np.std(x_square)
z = np.column_stack((np.ones(x.shape[0]), x_scaled, x_square_scaled))
j_2, theta_2 = gradientDescent_2(z,y,theta_init_2,alpha, iter_num)
print('Theta_2 tìm được là: ',np.round(theta_2,2))
for i in range (0,5):
    print(f'j_2 tìm được: {j_2[i]:.2f}')
# Dự đoán hàm linear
xtest1 = [1, 3.5]
xtest2 = [1, 7.0]
print(f'Giá trị dự đoán 1: {np.dot(xtest1, theta_1):.2f}')
print(f'Giá trị dự đoán 2: {np.dot(xtest2, theta_1):.2f}')
# Dự đoán hàm bậc 2
xtest3 = [1, 3.5, 12.25]
xtest4 = [1, 7.0, 49.0]
print(f'Giá trị dự đoán 1: {np.dot(xtest3, theta_2):.2f}')
print(f'Giá trị dự đoán 2: {np.dot(xtest4, theta_2):.2f}')
#vẽ hàm linear
x_plot = np.linspace(5,22.5,97)
plt.figure(0)
plt.scatter(xData, yData, c = 'red', marker = 'x', s = 20)
plt.plot(x[:,1], np.dot(x,theta_1), c = 'blue', lw = 2)
plt.plot(x_plot, np.dot(z,theta_2), c= 'green', lw = 2)
plt.xlabel("Population")
plt.ylabel("Profit")
plt.legend(['Data point', 'Hàm tuyến tính', 'Hàm bậc 2'], loc = 'upper left', bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()