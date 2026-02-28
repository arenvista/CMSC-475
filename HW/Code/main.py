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
    X = data.data[:, 0].reshape(-1, 1)
    y = data.target  # Labels (0 = malignant, 1 = benign)
    return X, y
    
def divide_dataset (data_x, data_y, train_percentage: float) :
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

def forward_pass(X: np.ndarray, w: np.ndarray, b: float):
    # Use dot product: (Samples, 30) @ (30,) -> (Samples,)
    z = np.dot(X, w) + b
    y_hat = 1 / (1 + np.exp(-z)) # Sigmoid
    return y_hat

def gradient_descent(w: np.ndarray, b: float, alpha: float, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray):
    num_entries = len(y)
    err = y_hat - y
    # Calculate average gradients
    grad_w = (1/num_entries) * np.dot(X.T, err) 
    grad_b = (1/num_entries) * np.sum(err)
    # Update
    new_w = w - (alpha * grad_w)
    new_b = b - (alpha * grad_b)
    return new_w, new_b

def loss(y:np.ndarray, y_hat: np.ndarray):
    num_ind: int = len(y)
    sq_err = lambda ind: (y_hat[ind] - y[ind])**2
    sum = 0
    for ind in range(len(y)):
        sum += sq_err(ind)
    mse = float((1/(2*num_ind))*sum)
    return mse

def plot_val(param: List[float], interval: int=1000, name: str="fig.png", title: str="title", xlabel: str="xlabel", ylabel: str="ylabel") -> None:
    itters = [i*interval for i in range(len(param))]
    
    # Set figsize to a square (e.g., 6x6 or 8x8)
    plt.figure(figsize=(6, 6))

    # Changed to (1, 1, 1) since there is only one plot. 
    # A 1x2 subplot in a square figure makes the graph look tall and skinny!
    plt.subplot(1, 1, 1) 
    
    plt.plot(itters, param, color='red', marker='o', linestyle='-', linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Using bbox_inches='tight' helps prevent labels from getting cut off
    plt.savefig(name, bbox_inches='tight') 
    plt.close() # Good practice to close the figure and free memory
    return
def print_evals(epoch, acc, loss):
    output = f""" Epoch [{epoch}] -----------
    \t Accuracy => {acc}
    \t Loss     => {loss}
    """
    print(output)

def train(x: np.ndarray, y: np.ndarray, alpha: float, num_itters: int):
    # Initialize weights as a vector of shape (num_features,)
    num_features = x.shape[1]
    weight = np.random.uniform(0.0, 1.0, size=num_features)
    bias = 0.0
    # Training loop ------
    interval: int  = 1000
    accu_vals = []
    loss_vals = []
    epochs = 0
    for i in range(num_itters): 
        y_hat = forward_pass(x, weight, bias)
        if (i%interval) == 0:
            # Record training accuracy after every 1000 iterations
            acc_val: float = calc_acc(y,y_hat)
            accu_vals.append(acc_val)
            # Record loss after every 1000 iterations 
            loss_val = loss(y,y_hat)
            loss_vals.append(loss_val)
            print_evals(epochs, acc_val, loss_val)
            epochs += 1
        weight, bias = gradient_descent(weight, bias, alpha, x,y,y_hat)
    # Plot acc vs iterations and loss vs iterations 
    plot_val(loss_vals, interval, "loss.png", "Loss v. Itter", "Itters", "Loss")
    plot_val(accu_vals, interval, "acc.png", "Accuracy v. Itters", "Itter", "Accuracy")
    return weight, bias 

def main():
    # 1. Load and Scale
    X, y = load_data()
    # 2. Split
    train_x, train_y, test_x, test_y = divide_dataset(X, y, 0.8)
    # 3. Train (Try alpha=0.1 for 10,000 iterations)
    print("Training started...")
    final_w, final_b = train(train_x, train_y, alpha=0.1, num_itters=10000)
    # 4. Evaluate on Test Set
    test_preds = forward_pass(test_x, final_w, final_b)
    final_acc = calc_acc(test_y, test_preds)
    print(f"Training Complete. Test Accuracy: {final_acc:.2%}")

if __name__ == "__main__":
    main()
