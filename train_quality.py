from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.metrics import r2_score
from keras.optimizers import SGD, RMSprop

class TrainQuality:
    def split_to_train_test_sets(self, wines):
        y = wines.quality
        x = wines.drop('quality', axis=1)
        x = StandardScaler().fit_transform(x)
        return x, y

    def get_model(self, x, y):
        seed = 7
        np.random.seed(seed)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        n = 0
        for train, test in kfold.split(x, y):
            n += 1
            # the first approach
            # model = Sequential()
            # model.add(Dense(64, input_dim=12, activation='relu'))
            # model.add(Dense(1))

            # add hidden layer
            # model = Sequential()
            # model.add(Dense(64, input_dim=12, activation='relu'))
            # model.add(Dense(64, activation='relu'))
            # model.add(Dense(1))

            # add hidden units
            model = Sequential()
            model.add(Dense(128, input_dim=12, activation='relu'))
            model.add(Dense(1))

            # first compile variant
            # model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

            # experiment with optimizer
            # rmsprop = RMSprop(lr=0.0001)
            # model.compile(optimizer=rmsprop, loss='mse', metrics=['mae'])
            sgd = SGD(lr=0.1)
            model.compile(optimizer=sgd, loss='mse', metrics=['mae'])

            model.fit(x[train], y[train], epochs=10, verbose=1)
            mse_value, mae_value = model.evaluate(x[test], y[test], verbose=0)
            print("fold number: " + str(n))
            print("mse = " + str(mse_value))
            print("mae = " + str(mae_value))
            y_pred = model.predict(x[test])
            # self.calc_additional(y[test], y_pred)

    def train(self, X_train, y_train):
        self.get_model(X_train, y_train)

    def calc_additional(self, y_test, y_pred):
        r2s = r2_score(y_test, y_pred)
        print("r2s = " + str(r2s))


