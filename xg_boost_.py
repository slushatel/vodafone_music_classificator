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
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

train_data = pd.read_csv("data/train_music.csv", sep=',')


# train_data.fillna(-99999, inplace=True)


def split_to_train_test_sets(data_set):
    x = data_set.ix[:, 2:data_set.shape[1]]
    y = np.ravel(data_set.target)
    return x, y


def standartize_data(x_train):
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    return x_train_s


def get_most_correlate_features(x, y):
    print("corellation")
    corframe = pd.DataFrame(columns=["FeatureNumber", "ColName", "Corr", "NullCount"])
    line_count = x.index.size
    n = -1
    for column in x:
        n = n + 1
        null_count = np.isnan(x[column].values).sum()
        if null_count / line_count > 0.01:
            continue
        nas = np.logical_or(np.isnan(x[column].values), np.isnan(y))
        cor = abs(scipy.stats.stats.pearsonr(x[column].values[~nas], y[~nas])[0])
        df = pd.DataFrame([[n, column, cor, null_count]], columns=["FeatureNumber", "ColName", "Corr", "NullCount"])
        corframe = corframe.append(df, ignore_index=True)
    corframe = corframe.sort_values(by="Corr", ascending=False)
    indices = []
    for i in range(20):
        print("column:" + str(corframe.values[i, 0]) + ":" + corframe.values[i, 1] + ", corr:" + str(corframe.values[
                                                                                                         i, 2]) + ", NullCount:" + str(
            corframe.values[i, 3]))
        indices.append(corframe.values[i, 0])
    return x.iloc[:, indices]


x, y = split_to_train_test_sets(train_data)
# x = get_most_correlate_features(x, y)
x_s = standartize_data(x)
x_train, x_test, y_train, y_test = train_test_split(x_s, y, test_size=0.4, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=51)

model_path = './vod_music_model.xgb'
print(os.path.abspath(model_path))


def f1_eval(y_pred, dtrain):
    # auc = metrics.roc_auc_score(y_test, y_pred)
    y_true = dtrain.get_label()
    # err = 1 - f1_score(y_true, np.round(y_pred))
    err = metrics.roc_auc_score(y_true, np.round(y_pred))
    return 'roc_auc', err


spw = (np.size(y_train) - np.count_nonzero(y_train)) / np.count_nonzero(y_train)
print("balance:" + str(spw))

dtrain = xgb.DMatrix(x_train, label=y_train, missing=np.NAN)
dval = xgb.DMatrix(x_val, label=y_val, missing=np.NAN)
dtest = xgb.DMatrix(x_test, label=y_test, missing=np.NAN)
param = {'max_depth': 5,
         'eta': 0.1,
         'verbosity': 3,
         # 'objective': 'binary:logistic',
         'objective': 'reg:linear',
         'n_estimators': 100,
         'nthread': -1,
         'eval_metric': ['auc'],
         'early_stopping_rounds': 5,
         # 'gamma': 10,
         'lambda ': 9,
         'subsample': 0.8,
         # 'min_child_weight': 10,
         'scale_pos_weight': spw}
evallist = [(dval, 'eval'), (dtrain, 'train')]
num_round = 50
model = xgb.train(param, dtrain, num_round, evallist, feval=f1_eval)

y_pred0 = model.predict(dtest)
# y_pred = [round(value) for value in y_pred0]
f = lambda x: 0 if x < 0.5 else 1
vfunc = np.vectorize(f)
y_pred = vfunc(y_pred0)

# model = XGBClassifier(objective="binary:logistic", max_depth=5, learning_rate=0.1, nthread=-1,
#                       scale_pos_weight=spw, n_estimators=100, verbosity=3)
# model.fit(x_train, y_train, verbose=True, eval_metric="aucpr")
# y_pred = model.predict(x_test)
# y_pred = [round(value) for value in y_pred0]

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
f1 = f1_score(y_test, y_pred)
print("F1: %.2f%%" % (f1 * 100.0))

plt.figure(1)
print(confusion_matrix(y_test, y_pred))
plt.show()

plt.figure(2)
xgb.plot_importance(model, max_num_features=15)
plt.show()

os.environ["PATH"] += os.pathsep + 'c:/Program Files (x86)/Graphviz2.38/bin/'
plt.figure(3)
xgb.plot_tree(model, num_trees=2)
plt.show()

cor_test = scipy.stats.stats.pearsonr(y_test, y_pred)[0]
print("Correlation: %.2f%%" % (cor_test * 100.0))
plt.figure(4)
vis.Vizualizer().plot_distribution(y_test)
plt.figure(5)
vis.Vizualizer().plot_distribution(y_pred)
# vis.Vizualizer().plot_distribution(y_pred0)

plt.figure(6)
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("ROC AUC: " + str(auc))
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()
