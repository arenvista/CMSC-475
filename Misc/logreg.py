import numpy as np
def gen_matrix(matrix):
    total="\\begin{bmatrix}\n"
    for row in matrix:
        row_str = " & ".join([str(i) for i in row ]) + " \\\\\n"
        total += row_str
    total+="\\end{bmatrix}"
    return total


def sigmoid(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X))

def loss(y: np.ndarray, y_hat: np.ndarray):
    num_ind = len(y)
    mse = (1 / (2 * num_ind)) * np.sum((y_hat - y) ** 2)
    return float(mse)

def fp(X, w, b: float):
    z = np.dot(X, w) + b
    return sigmoid(z)

def gradient_descent(w, b: float, alpha: float, X, y):
    # Forward pass
    y_hat = fp(X, w, b)
    print(y_hat)
    
    # Error
    err = y_hat - y
    # print("err_grad")
    # print(err)
    
    # Sigmoid derivative
    sigmoid_grad = y_hat * (1 - y_hat)
    # print("sig_grad")
    # print(sigmoid_grad)
    
    # Correct gradient for MSE + sigmoid
    grad_w = np.dot(X.T, err * sigmoid_grad)
    # print("grad_w")
    # print(grad_w)
    print(gen_matrix(grad_w))
    
    # Update
    new_w = w - alpha * grad_w
    
    return new_w

# ----- Data -----
case = 1
if case==1:
    X = np.array([[1, 2, 1]])   # second example
    y = np.array([[0]])         # correct shape (1x1)
    w = np.array([[-2], [2], [1]])  # column vector

# SGD step
    new_w = gradient_descent(w, 0, 1.0, X, y)

    print("Updated weights:\n", new_w)
    print(gen_matrix(new_w))

# ----- Data -----
if case==2:
    X = np.array([[1, 1, -1]])   # second example
    y = np.array([[1]])         # correct shape (1x1)
    w = np.array([[-2], [2], [1]])  # column vector

# SGD step
    new_w = gradient_descent(w, 0, 1.0, X, y)

    print("Updated weights:\n", new_w)
    print(gen_matrix(new_w))
