import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from typing import List

data = pd.read_csv("C:/Users/nikmi/Diamonds/diamonds.csv")
y = data['price']
X = data.drop('price', axis=1)


def view(dat: pd.DataFrame, num: int = None):
    print(dat.head(num))


def delete_useless_col(*column: List[str], our_data):
    for i in column:
        our_data.drop(our_data[i], axis=1, inplace=True)
    return our_data


def information(our_data: pd.DataFrame):
    return our_data.info()


def statistical_info(our_data: pd.DataFrame):
    return our_data.describe()


def correlation(our_data: pd.DataFrame):
    return our_data.corr()


def histogram():
    for i in range(len(X.describe().keys()) + 1):
        if type(X.iloc[i][i]) == str:
            continue
        else:
            plt.figure(figsize=(20, 10))
            plt.hist(X.iloc[:, i], bins=100)
            plt.show()


def encoder(columns: List[str], enc):
    encod = enc()
    for i in columns:
        X[i] = encod.fit_transform(X[i].values.reshape(-1, 1))


def training(model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    mod = model()
    mod.fit(X_train, y_train)
    print("Score: {:.5f}%".format(mod.score(X_test, y_test) * 100))


