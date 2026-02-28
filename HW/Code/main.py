from numpy._core.numerictypes import float64
from sklearn.datasets import load_breast_cancer
import numpy as np 
import numpy.random as random
import pandas as pd
from numpy._core.numerictypes import float64
from numpy.typing import NDArray
import math
from typing import Any, List
import matplotlib.pyplot as plt

def load_data():
    data: Any = load_breast_cancer()
    # Access data and targets
    X = data.data    # Features (569 samples, 30 features)
    y = data.target  # Labels (0 = malignant, 1 = benign)
    return X, y
    
def divide_dataset (data_x , data_y , train_percentage ) :
    n_y = int(len(data_y)*train_percentage)
    n_x = int(len(data_x)*train_percentage)
    train_data_x = data_x[0:n_x]  
    train_data_y = data_y[0:n_y]  
    test_data_x  = data_x[n_x:]   
    test_data_y  = data_y[n_y:]  
    return train_data_x , train_data_y , test_data_x , test_data_y

def sigmoid_vec(X: np.ndarray):
    sigmoid_func = lambda x: 1 / (1 + math.exp(-x))
    for x_vec in X:
        for x_ind in range(len(x_vec)):
            X[x_ind] = sigmoid_func(X[x_ind])

def calc_acc(y: np.ndarray, y_hat: np.ndarray):
    round_val = lambda val: 1 if val > 0.5 else 0
    num_entries = len(y); corr = 0
    for i in range(len(y)):
        y_hat_round = round_val(y_hat[i])
        if y[i] == y_hat_round: corr += 1
    return corr / num_entries

def ini_param(num_entries: int, lower_bound: float=0.0, upper_bound: float=1.0) -> NDArray[float64]:
    param = []
    for _ in range(num_entries):
        x = random.uniform(lower_bound, upper_bound)
        param.append(x)
    return np.array(param)

def forward_pass(x: np.ndarray, w: float, b: float):
    # Forward Pass
    y_hat: NDArray[float64] = (w*x)+b
    return y_hat

def gradient_descent(w: float, b: float, alpha: float, x: np.ndarray, y: np.ndarray, y_hat: np.ndarray):
    # Grads 
    err: NDArray[float64] = y_hat -y
    grad_w: NDArray[float64] = err * x
    grad_b: NDArray[float64] = err

    # Update Params
    new_w = w - (alpha*grad_w)
    new_b = b - (alpha*grad_b)
    return new_w, new_b

def loss(y:np.ndarray, y_hat: np.ndarray):
    num_ind: int = len(y)
    sq_err = lambda ind: (y_hat[ind] - y[ind])**2
    sum = 0
    for ind in range(len(y)):
        sum += sq_err(ind)
    mse = float((1/(2*num_ind))*sum)
    return mse

def plot_val(param: List[float], interval: int=1000):
    itters = [i*interval for i in range(len(param))]
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(itters, param, color='red', marker='o', linestyle='-', linewidth=2)
    plt.title('Loss vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Cross-Entropy)')
    plt.grid(True, linestyle='--', alpha=0.6)
    return

def  train(x:np.ndarray, y:np.ndarray, alpha: float, num_itters: int):
    # Initialize params 
    weight: float = random.uniform(0.0,1.0)
    bias: float = random.uniform(0.0,1.0)
    # Training loop ------
    interval: int  = 1000
    accu_vals = []
    loss_vals = []
    for i in range(num_itters): 
        y_hat = forward_pass(x, weight, bias)
        if (i%interval) == 0:
            # Record training accuracy after every 1000 iterations
            acc: float = calc_acc(y,y_hat)
            accu_vals.append(acc)
            # Record loss after every 1000 iterations 
            loss_val = loss(y,y_hat)
            loss_vals.append(loss_val)
        weight, bias = gradient_descent(weight, bias, alpha, x,y,y_hat)
    # Plot acc vs iterations and loss vs iterations 
    plot_val(loss_vals,interval)
    plot_val(accu_vals,interval)
    final_m=0; final_b=0 
    return final_m, final_b 
def main():
    X, y = load_data()
    divide_dataset(X,y, 0.8)
    sigmoid_vec(X[0])
    a = ini_param(5)
    print(a)

if __name__ == "__main__":
    main()
