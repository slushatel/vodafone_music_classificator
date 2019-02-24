# Import pandas
import pandas as pd
from sklearn.preprocessing import StandardScaler

import train_quality
import visualizer as vis
import numpy as np
import train as train
from sklearn.metrics import r2_score
import scipy

# Read in white wine data
train_data = pd.read_csv("d:/Downloads/Vodafone Music Challenge/Data/train_music.csv", sep=',')

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
print("corellation")
# corlist = []
corframe = pd.DataFrame(columns=["ColName", "Corr", "NullCount", "Mean"])
line_count = x_train.index.size

for column in x_train:
    nas = np.logical_or(np.isnan(x_train[column].values), np.isnan(y_train))
    cor = abs(scipy.stats.stats.pearsonr(x_train[column].values[~nas], y_train[~nas])[0])
    print("corellation of " + column + ":" + str(cor))
    null_count = np.isnan(x_train[column].values).sum()
    if (null_count / line_count > 0.1):
        continue
    # mean = x_train[column].values[~nas].mean()
    mean = x_train.loc[:,column].mean()
    df = pd.DataFrame([[column, cor, null_count, mean]], columns=["ColName", "Corr", "NullCount", "Mean"])
    corframe = corframe.append(df, ignore_index=True)

corframe = corframe.sort_values(by="Corr", ascending=False)

# fill nan with mean
df_mean = x_train.mean()
x_train.fillna(df_mean, inplace=True)
x_test.fillna(df_mean, inplace=True)



"""
the count of most important features we will use
"""
n_features = 4

#
# X_train, X_test = train.Trainer().standartize_data(X_train, X_test)
#
model = train.Trainer().train(x_train[corframe.values[0:n_features, 0]], y_train)
#
y_pred = model.predict(x_test[corframe.values[0:n_features, 0]])
# vis.Vizualizer().plot_distribution(y_pred.round())
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
