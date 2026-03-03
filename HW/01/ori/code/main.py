from numpy._core.numerictypes import float64
from datetime import datetime
import glob
from sklearn.datasets import load_breast_cancer
import numpy as np
import numpy.random as random
import pandas as pd
from numpy._core.numerictypes import float64
from numpy.typing import NDArray
import math
from typing import Any, List, Tuple
import matplotlib.pyplot as plt
import argparse

LOG_MODE: bool = False
VERBOSE_MODE: bool = False
ALPHA: float = 0.0

def load_data() -> Tuple[NDArray, NDArray]:
    data: Any = load_breast_cancer()
    X: NDArray = data.data[:, 0].reshape(-1, 1)
    y: NDArray = data.target
    return X, y

def divide_dataset(
    data_x: NDArray, data_y: NDArray, train_percentage: float
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    n_entries: int = int(len(data_y) * train_percentage)
    train_data_x: NDArray = data_x[0:n_entries]
    train_data_y: NDArray = data_y[0:n_entries]
    test_data_x: NDArray = data_x[n_entries:]
    test_data_y: NDArray = data_y[n_entries:]
    return train_data_x, train_data_y, test_data_x, test_data_y


def sigmoid_vec(X: np.ndarray) -> None:
    sigmoid_func = lambda x: 1 / (1 + math.exp(-x))
    for x_vec in X:
        for x_ind in range(len(x_vec)):
            X[x_ind] = sigmoid_func(X[x_ind])


def calc_acc(y: np.ndarray, y_hat: np.ndarray) -> float:
    round_val = lambda val: 1 if val > 0.5 else 0
    num_entries = len(y)
    corr = 0
    for i in range(len(y)):
        y_hat_round = round_val(y_hat[i])
        if y[i] == y_hat_round:
            corr += 1
    return corr / num_entries


def forward_pass(X: np.ndarray, w: np.ndarray, b: float):
    # Use dot product: (Samples, 30) @ (30,) -> (Samples,)
    z = np.dot(X, w) + b
    y_hat = 1 / (1 + np.exp(-z))  # Sigmoid
    return y_hat


def gradient_descent(
    w: np.ndarray,
    b: float,
    alpha: float,
    X: np.ndarray,
    y: np.ndarray,
    y_hat: np.ndarray,
):
    num_entries = len(y)
    err = y_hat - y
    # Calculate average gradients
    grad_w = (1 / num_entries) * np.dot(X.T, err)
    grad_b = (1 / num_entries) * np.sum(err)
    # Update
    new_w = w - (alpha * grad_w)
    new_b = b - (alpha * grad_b)
    return new_w, new_b

def loss(y: np.ndarray, y_hat: np.ndarray):
    num_ind: int = len(y)
    sq_err = lambda ind: (y_hat[ind] - y[ind]) ** 2
    sum = 0
    for ind in range(len(y)):
        sum += sq_err(ind)
    mse = float((1 / (2 * num_ind)) * sum)
    return mse

def plot_val(
    param: List[float],
    interval: int = 1000,
    name: str = "fig.png",
    title: str = "title",
    xlabel: str = "xlabel",
    ylabel: str = "ylabel",
) -> None:
    itters = [i * interval for i in range(len(param))]

    # Set figsize to a square (e.g., 6x6 or 8x8)
    plt.figure(figsize=(6, 6))

    # Changed to (1, 1, 1) since there is only one plot.
    # A 1x2 subplot in a square figure makes the graph look tall and skinny!
    plt.subplot(1, 1, 1)

    plt.plot(itters, param, color="red", marker="o", linestyle="-", linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Using bbox_inches='tight' helps prevent labels from getting cut off
    plt.savefig(name, bbox_inches="tight")
    plt.close()  # Good practice to close the figure and free memory
    return

def print_evals(epoch, acc, loss):
    if VERBOSE_MODE:
        output = f""" Epoch [{epoch}] -----------\n \t Accuracy => {acc}\n \t Loss     => {loss} """
        print(output)

def gen_csv_filename():
    now = datetime.now()
    timestamp_id = now.strftime("%Y%m%d_%H%M%S")
    filename = f"log/log_{timestamp_id}.csv"
    return filename


def write_logs(acc_vals, interval, alpha):
    # Output logs in csv format
    header = ["alpha"] + [f"N{i}({i*interval})" for i in range(len(acc_vals))]
    header_str = ",".join(header)
    acc_str = ",".join([f"{ac:.5f}" for ac in acc_vals])
    out_str = f"{header_str}\n{alpha},{acc_str}"
    log_filename = gen_csv_filename()
    with open(log_filename, "w") as file:
        file.write(out_str)

def write_pretty_table(filename: str):
    file_list = glob.glob("log/*.csv")
    dfs = [pd.read_csv(f) for f in file_list]
    master_df = pd.concat(dfs, ignore_index=True)
    headers = master_df.columns.tolist()
    data = master_df.values.tolist()
    pretty_table(data, headers)


def pretty_table(data, headers):
    col_width = 12
    out_strs = []
    sep = "+" + ("-" * (col_width + 2) + "+") * len(headers)
    header_row = "| " + " | ".join(f"{h:^{col_width}}" for h in headers) + " |"
    out_strs.append(sep)
    out_strs.append(header_row)
    out_strs.append(sep.replace("-", "="))
    for row in data:
        row_str = "| " + " | ".join(f"{str(item):^{col_width}}" for item in row) + " |"
        out_strs.append(row_str)
    out_strs.append(sep)
    with open("tbl.md", "w") as f:
        f.write("\n".join(out_strs))


def train(x: np.ndarray, y: np.ndarray, alpha: float, num_itters: int):
    # Initialize weights as a vector of shape (num_features,)
    num_features = x.shape[1]
    weight = np.random.uniform(0.0, 1.0, size=num_features)
    bias = 0.0
    # Training loop ------
    interval: int = 1000
    accu_vals = []
    loss_vals = []
    epochs = 0
    for i in range(num_itters):
        y_hat = forward_pass(x, weight, bias)
        if (i % interval) == 0:
            # Record training accuracy after every 1000 iterations
            acc_val: float = calc_acc(y, y_hat)
            accu_vals.append(acc_val)
            # Record loss after every 1000 iterations
            loss_val = loss(y, y_hat)
            loss_vals.append(loss_val)
            print_evals(epochs, acc_val, loss_val)
            epochs += 1
        weight, bias = gradient_descent(weight, bias, alpha, x, y, y_hat)
    # Plot acc vs iterations and loss vs iterations
    plot_val(loss_vals, interval, "loss.png", "Loss v. Itter", "Itters", "Loss")
    plot_val(accu_vals, interval, "acc.png", "Accuracy v. Itters", "Itter", "Accuracy")
    if LOG_MODE:
        write_logs(accu_vals, interval, alpha)
        write_pretty_table("table.md")
    return weight, bias


def test(test_x: np.ndarray, test_y: np.ndarray, weight: NDArray, bias: float):
    y_hat = forward_pass(test_x, weight, bias)
    final_acc = calc_acc(test_y, y_hat)
    print(f"Training Complete. Fin Accuracy: {final_acc:.2%}")


def ini_parse():
    parser = argparse.ArgumentParser(description="Breast Cancer Neural Network")
    parser.add_argument(
        "-a", "--alpha", type=float, default=0.1, help="The learning rate (alpha)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument("-l", "--log", action="store_true", help="enable log")
    args = parser.parse_args()
    global LOG_MODE, VERBOSE_MODE, ALPHA
    LOG_MODE = args.log
    VERBOSE_MODE = args.verbose
    ALPHA = args.alpha
    return args


def main():
    args = ini_parse()
    alpha: float = args.alpha
    X, y = load_data()
    num_itters = 10000
    train_x, train_y, test_x, test_y = divide_dataset(X, y, 0.8)
    print("Training started...")
    final_w, final_b = train(train_x, train_y, alpha, num_itters)
    test(test_x, test_y, final_w, final_b)


if __name__ == "__main__":
    main()
