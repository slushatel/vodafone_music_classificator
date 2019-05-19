import pandas as pd
import visualizer as vis
import numpy as np
import train as train
from sklearn.metrics import r2_score
import scipy
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics

# Read in white wine data
train_data = pd.read_csv("data/train_music.csv", sep=',')

# undersample 0 class
# no_frauds = len(train_data[train_data['target'] == 1])
# non_fraud_indices = train_data[train_data.target == 0].index
# random_indices = np.random.choice(non_fraud_indices, no_frauds, replace=False)
# fraud_indices = train_data[train_data.target == 1].index
# under_sample_indices = np.concatenate([fraud_indices, random_indices])
# train_data = train_data.loc[under_sample_indices]

train_data.fillna(train_data.mean(), inplace=True)

x_train, x_test, y_train, y_test = train.Trainer().split_to_train_test_sets(train_data)
clf = LogisticRegression(penalty='l2', C=1, class_weight='balanced', solver='liblinear')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

cor_test = scipy.stats.stats.pearsonr(y_test * 2 - 1, y_pred * 2 - 1)[0]
print("correlation", cor_test)

print("Accuracy", metrics.accuracy_score(y_test, y_pred))
y_pred_proba = clf.predict_proba(x_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

# print("corellation")
# corframe = pd.DataFrame(columns=["ColName", "Corr", "NullCount", "Mean"])
# line_count = x_train.index.size
# for column in x_train:
#     nas = np.logical_or(np.isnan(x_train[column].values), np.isnan(y_train))
#     cor = abs(scipy.stats.stats.pearsonr(x_train[column].values[~nas], y_train[~nas])[0])
#     print("corellation of " + column + ":" + str(cor))
#     null_count = np.isnan(x_train[column].values).sum()
#     if (null_count / line_count > 0.1):
#         continue
#     # mean = x_train[column].values[~nas].mean()
#     mean = x_train.loc[:, column].mean()
#     df = pd.DataFrame([[column, cor, null_count, mean]], columns=["ColName", "Corr", "NullCount", "Mean"])
#     corframe = corframe.append(df, ignore_index=True)
# corframe = corframe.sort_values(by="Corr", ascending=False)
#
# # fill nan with mean
# df_mean = x_train.mean()
# x_train.fillna(df_mean, inplace=True)
# x_test.fillna(df_mean, inplace=True)

"""
the count of most important features we will use
"""
# n_features = 4

# model = train.Trainer().train(x_train[corframe.values[0:n_features, 0]], y_train)
# y_pred = model.predict(x_test[corframe.values[0:n_features, 0]])
# vis.Vizualizer().plot_distribution(np.transpose(y_pred)[0])
#
# cor_test = scipy.stats.stats.pearsonr(y_test, np.transpose(y_pred.round())[0])[0]
#
# score = model.evaluate(x_test[corframe.values[0:n_features, 0]], y_test, verbose=1)
# print(score)
