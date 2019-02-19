# Import pandas
import pandas as pd
from sklearn.preprocessing import StandardScaler

import train_quality
import visualizer as vis
import numpy as np
import train as train
from sklearn.metrics import r2_score

# Read in white wine data
train_data = pd.read_csv("c:/Downloads/Vodafone Music Challenge/Data/train_music.csv", sep=',')

# Read in red wine data
# red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

# Print info on white wine
print(train_data.info())

# Print info on red wine
# print(red.info())

# First rows of `red`
# print(red.head())

# Last rows of `white`
# print(white.tail())

# Take a sample of 5 rows of `red`
print(train_data.sample(5))

# Describe `white`
print(train_data.describe())

# Double check for null values in `red`
print(pd.isnull(train_data))

# too long
# vis.Vizualizer().plot_corelation_matrix(train_data)

###########################################

x_train, x_test, y_train, y_test = train.Trainer().split_to_train_test_sets(train_data)
#
# X_train, X_test = train.Trainer().standartize_data(X_train, X_test)
#
model = train.Trainer().train(x_train, y_train)
#
y_pred = model.predict(x_test)
print(y_pred[:5])
print(y_test[:5])
#
score = model.evaluate(x_test, y_test, verbose=1)
print(score)
#
# train.Trainer().calc_additional(y_test, y_pred.round())
#
# ###########################################
#
# x_q, y_q = train_quality.TrainQuality().split_to_train_test_sets(train_data)
# train_quality.TrainQuality().train(x_q, y_q)
