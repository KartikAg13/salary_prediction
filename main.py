import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

df = pd.read_csv('Salary_dataset.csv')

x_train = df['YearsExperience'].to_numpy(dtype=float)
y_train = df['Salary'].to_numpy(dtype=int)

y_train_shape = y_train.shape
print(f"Shape of data: {y_train_shape}")

m = len(x_train)
print(f"Length of data: {m}")

w_initial = 100
b_initial = 100

def compute_cost(x, y, w, b):
    f_wb = (w * x) + b
    cost = np.sum(np.square(f_wb - y)) / (2 * m)
    return cost, f_wb

cost_initial, f_wb = compute_cost(x_train, y_train, w_initial, b_initial)
print(f"Initial cost: {cost_initial}")

def draw_function(x, y, f_wb):
    plt.plot(x, f_wb, label="Prediction", c='b')
    plt.scatter(x, y, label="Actual", marker='x', c='r')
    plt.title('Salary Data')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()

draw_function(x_train, y_train, f_wb)
