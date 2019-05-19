import pandas as pd
import numpy as np
import visualizer as vis
import train as train
from sklearn.metrics import r2_score
import scipy
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Read in white wine data
train_data = pd.read_csv("data/train_music.csv", sep=',')
train_data.fillna(-999, inplace=True)


# print(train_data.info())
# print(train_data.sample(5))
# print(train_data.describe())
# print(pd.isnull(train_data))

# undersample 0 class
# no_frauds = len(train_data[train_data['target'] == 1])
# non_fraud_indices = train_data[train_data.target == 0].index
# random_indices = np.random.choice(non_fraud_indices, no_frauds, replace=False)
# fraud_indices = train_data[train_data.target == 1].index
# under_sample_indices = np.concatenate([fraud_indices, random_indices])
# train_data = train_data.loc[under_sample_indices]

def split_to_train_test_sets(data_set):
    x = data_set.ix[:, 2:data_set.shape[1]]
    y = np.ravel(data_set.target)
    return x, y


def standartize_data(x_train):
    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    return x_train_s


x, y = split_to_train_test_sets(train_data)
# x = get_most_correlate_features(x, y)
x_s = standartize_data(x)
x_train, x_test, y_train, y_test = train_test_split(x_s, y, test_size=0.2, random_state=42)
# x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=51)

# x_train, x_test, y_train, y_test = train.Trainer().split_to_train_test_sets(train_data)
print("corellation")
corframe = pd.DataFrame(columns=["ColName", "Corr", "NullCount", "Mean"])
line_count = x_train.shape[0]
indices = np.empty((0, 2), int)
for col in range(x_train.shape[1]):
    # nas = np.logical_or(np.isnan(x_train[column].values), np.isnan())
    # cor = abs(scipy.stats.stats.pearsonr(x_train[column].values[~nas], y_train[~nas])[0])
    cor = abs(scipy.stats.stats.pearsonr(x_train[:, col], y_train)[0])
    print("corellation of column number - " + str(col) + ":" + str(cor))
    # null_count = np.isnan(x_train[column].values).sum()
    # if (null_count / line_count > 0.1):
    #     continue
    # mean = x_train[column].values[~nas].mean()
    # mean = x_train.loc[:, column].mean()
    indices = np.vstack([indices, [col, cor]])
    # df = pd.DataFrame([[column, cor, null_count, mean]], columns=["ColName", "Corr", "NullCount", "Mean"])
    # corframe = corframe.append(df, ignore_index=True)
# corframe = corframe.sort_values(by="Corr", ascending=False)
indices = indices[np.argsort(indices[:, 1])]
indices = np.flip(indices, 0)
# fill nan with mean
# df_mean = x_train.mean()
# x_train.fillna(df_mean, inplace=True)
# x_test.fillna(df_mean, inplace=True)

"""
the count of most important features we will use
"""
n_features = 20
ind_list = indices[0:n_features, 0].astype(int)

# model = train.Trainer().train(x_train[corframe.values[0:n_features, 0]], y_train)
model = train.Trainer().train(x_train.T[ind_list].T, y_train)
# y_pred = model.predict(x_test[corframe.values[0:n_features, 0]])
y_pred = model.predict(x_test.T[ind_list].T)
vis.Vizualizer().plot_distribution(np.transpose(y_pred)[0])

# cor_test = scipy.stats.stats.pearsonr(y_test, np.transpose(y_pred.round())[0])[0]
# score = model.evaluate(x_test[corframe.values[0:n_features, 0]], y_test, verbose=1)
# print(score)

plt.figure(6)
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
print("ROC AUC: " + str(auc))
plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
plt.legend(loc=4)
plt.show()
