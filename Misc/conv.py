import numpy as np
def gen_matrix(matrix):
    total="\\begin{bmatrix}\n"
    for row in matrix:
        row_str = " & ".join([str(i) for i in row ]) + " \\\\\n"
        total += row_str
    total+="\\end{bmatrix}"
    print(total)

# Input matrix x (3x2)
x = np.array([[1, 10],
              [2, 20],
              [3, 30]])

# Kernel h (2x3)
h = np.array([[1, 0, 1],
              [-1, 0, -1]])

# Flip the kernel for true convolution
h_flipped = np.flip(np.flip(h, axis=0), axis=1)

# Sizes
x_rows, x_cols = x.shape
h_rows, h_cols = h_flipped.shape

# Zero padding
pad_height = h_rows - 1
pad_width = h_cols - 1
x_padded = np.pad(x, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

# Output size
out_rows = x_rows + h_rows - 1
out_cols = x_cols + h_cols - 1
y = np.zeros((out_rows, out_cols))

# Perform convolution with detailed printout
print("Flipped kernel (used for convolution):")
print(h_flipped)
print("\n--- Convolution steps ---\n")

cont = 0
for i in range(out_rows):
    for j in range(out_cols):
        print("\[")
        patch = x_padded[i:i+h_rows, j:j+h_cols]
        conv_value = np.sum(patch * h_flipped)
        y[i, j] = conv_value
        # print(f"Patch at output position ({i},{j}):")
        # print(patch)
        gen_matrix(patch)
        gen_matrix(h_flipped)
        # print("Element-wise multiply with kernel:")
        # print(patch * h_flipped)
        print(f"= {conv_value} = a_"+ '{' + f"({i},{j})" + '}')
        print("\]\n")

print("\n\nFinal convolution result:--------------------")
gen_matrix(y)

gen_matrix(x_padded)
