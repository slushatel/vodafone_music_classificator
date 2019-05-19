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
from numpy import array

train_data = pd.read_csv("data/train_music.csv", sep=',')
train_data.fillna(-9999999, inplace=True)

x_train, x_test, y_train, y_test = train.Trainer().split_to_train_test_sets(train_data)

print("corellation")
corframe = pd.DataFrame(columns=["FeatureNumber", "ColName", "Corr", "NullCount"])
line_count = x_train.index.size
n = 0
for column in x_train:
    nas = np.logical_or(np.isnan(x_train[column].values), np.isnan(y_train))
    cor = abs(scipy.stats.stats.pearsonr(x_train[column].values[~nas], y_train[~nas])[0])
    print("corellation of " + column + ":" + str(cor))
    null_count = np.isnan(x_train[column].values).sum()
    if (null_count / line_count > 0.1):
        continue
    df = pd.DataFrame([[n, column, cor, null_count]], columns=["FeatureNumber", "ColName", "Corr", "NullCount"])
    corframe = corframe.append(df, ignore_index=True)
    n = n + 1

corframe = corframe.sort_values(by="Corr", ascending=False)

color = ['orange' if l == 0 else 'black' for l in y_train]
s = [1 if l == 0 else 2 for l in y_train]


def not_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)


for i in range(10):
    print("column:" + str(corframe.values[i, 0]) + ":" + corframe.values[i, 1] + ", corr:" + str(corframe.values[
        i, 2]) + ", NullCount:" + str(corframe.values[i, 3]))

    # n = 0
# for i in range(6):
#     for j in range(6):
#         n = n + 1
#
#         not_outliers_ = np.logical_and(not_outliers(x_train[corframe.values[i, 0]]), not_outliers(x_train[corframe.values[j, 0]]))
#         print("columns:" + corframe.values[i, 0] + ", " + corframe.values[j, 0])
#         plt.figure(n)
#         plt.scatter(x_train[corframe.values[i, 0]][not_outliers_], x_train[corframe.values[j, 0]][not_outliers_],
#                     color=array(color)[not_outliers_], s=array(s)[not_outliers_])
#         # plt.interactive(False)
#         plt.title(corframe.values[i, 0] + ", " + corframe.values[j, 0]);
#         plt.show()

print("Finished")
