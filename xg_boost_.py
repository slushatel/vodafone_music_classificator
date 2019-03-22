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


def split_to_train_test_sets(data_set):
    x = data_set.ix[:, 2:data_set.shape[1]]
    y = np.ravel(data_set.target)
    return x, y


def standartize_data(x_train):
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    return x_train_s


x, y = split_to_train_test_sets(train_data)
x_s = standartize_data(x)
x_train, x_test, y_train, y_test = train_test_split(x_s, y, test_size=0.2, random_state=42)

model_path = './vod_music_model.xgb'
print(os.path.abspath(model_path))
# if os.path.isfile(model_path):
#     model = xgboost.load_model(model_path)
# else:

spw = (np.size(y_train) - np.count_nonzero(y_train)) / np.count_nonzero(y_train)
model = XGBClassifier(nthread=-1, scale_pos_weight=spw, n_estimators=100, silent=False)
model.fit(x_train, y_train, verbose=True)
# xgboost.save_model(model_path, model)

# make predictions for test data
y_pred = model.predict(x_test)
# y_pred = [round(value) for value in y_pred0]
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

plt.figure(figsize=(20, 15))
xgb.plot_importance(model, ax=plt.gca())

os.environ["PATH"] += os.pathsep + 'c:/Program Files (x86)/Graphviz2.38/bin/'
plt.figure(figsize=(20, 15))
xgb.plot_tree(model, ax=plt.gca())

print("Number of boosting trees: {}".format(model.n_estimators))
print("Max depth of trees: {}".format(model.max_depth))
print("Objective function: {}".format(model.objective))

cor_test = scipy.stats.stats.pearsonr(y_test, y_pred)[0]
print("Correlation: %.2f%%" % (cor_test * 100.0))
vis.Vizualizer().plot_distribution(y_test)
vis.Vizualizer().plot_distribution(y_pred)
# vis.Vizualizer().plot_distribution(y_pred0)

fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("ROC AUC: " + str(auc))
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()

