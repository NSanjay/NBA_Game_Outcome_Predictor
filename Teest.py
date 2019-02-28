import getData
from sklearn import svm
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def doTest():
    X = []
    Y = []
    trainError = []
    i = 1
    teams = []
    csv_files = df.read_csv('../dataset/data.csv',header=0)
    print(csv_files.head())

def main():
    doTest()

if __name__ == '__main__':
    main()