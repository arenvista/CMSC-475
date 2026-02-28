import pandas as pd
from sklearn.datasets import load_breast_cancer

df, y = load_breast_cancer(as_frame=True, return_X_y=True)

def main():
    print("Hello from code!")
    print(df)
    print("----")
    # print(y)
    divide_dataset(y,y, 0.8)

def divide_dataset ( data_x , data_y , train_percentage ) :
    n_y = len(data_y)*train_percentage
    n_x = len(data_x)*train_percentage
    train_data_x = data_x[0:n_x]  
    train_data_y = data_y[0:n_y]  
    test_data_x  = data_x[n_x:]   
    test_data_y  = data_y[n_y:]  

    print(f"train_data_x => {len(train_data_x)} ")
    print(f"train_data_y => {len(train_data_y)} ")
    print(f"test_data_x => {len(test_data_x)} ")
    print(f"test_data_y => {len(test_data_y)} ")
    return train_data_x , train_data_y , test_data_x , test_data_y

if __name__ == "__main__":
    main()
