import pandas as pd
import numpy as np
import train as train
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy
import os
import visualizer as vis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

train_data = pd.read_csv("data/train_music.csv", sep=',')

x_train, x_test, y_train, y_test = train.Trainer().split_to_train_test_sets(train_data)

print("corellation")
corframe = pd.DataFrame(columns=["ColName", "Corr", "NullCount", "Mean"])
line_count = x_train.index.size
for column in x_train:
    nas = np.logical_or(np.isnan(x_train[column].values), np.isnan(y_train))
    cor = abs(scipy.stats.stats.pearsonr(x_train[column].values[~nas], y_train[~nas])[0])
    print("corellation of " + column + ":" + str(cor))
    null_count = np.isnan(x_train[column].values).sum()
    if (null_count / line_count > 0.1):
        continue
    df = pd.DataFrame([[column, cor, null_count]], columns=["ColName", "Corr", "NullCount"])
    corframe = corframe.append(df, ignore_index=True)

corframe = corframe.sort_values(by="Corr", ascending=False)

color = ['red' if l == 0 else 'green' for l in y_train]

n = 0
for i in range(4):
    for j in range(4):
        n = n + 1

        print("columns:" + corframe.values[i, 0] + ", " + corframe.values[j, 0])
        plt.figure(n)
        plt.scatter(x_train[corframe.values[i, 0]], x_train[corframe.values[j, 0]], color=color)
        plt.interactive(False)
        plt.show()

print("Finished")
