import pandas as pd

if __name__ == "__main__":
    problem_csv = pd.read_csv("/home/liubanruo/test_data/data229/1094660/1094660_SC_normalized.csv", header=None)
    print(problem_csv.shape)
    print(problem_csv.head())
    print("___________")
    print(problem_csv.tail())
    import numpy as np
    problem_csv = np.empty(())
    print(problem_csv.shape)
    problem_csv = np.zeros((3,4))
    print(problem_csv.shape)
    if problem_csv.shape != (3,3):
        print(True)


