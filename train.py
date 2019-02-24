from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import os
import keras.models
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score


class Trainer:
    def split_to_train_test_sets(self, wines):
        # Specify the data
        X = wines.ix[:, 2:wines.shape[1]]

        # Specify the target labels and flatten the array
        y = np.ravel(wines.target)

        # Split the data up in train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def standartize_data(self, X_train, X_test):
        # Define the scaler
        scaler = StandardScaler().fit(X_train)

        # Scale the train set
        X_train = scaler.transform(X_train)

        # Scale the test set
        X_test = scaler.transform(X_test)

        return X_train, X_test

    def get_model(self, x_train, y_train):
        model_path = './vod_music_model.h5'
        print(os.path.abspath(model_path))
        if os.path.isfile(model_path):
            model = keras.models.load_model(model_path);
        else:
            model = Sequential()

            # Add an input layer
            model.add(Dense(x_train.shape[1], activation='relu', input_shape=(x_train.shape[1],)))

            # Add one hidden layer
            model.add(Dense(round((x_train.shape[1] + 1) / 2), activation='relu'))

            # Add an output layer
            model.add(Dense(1, activation='sigmoid'))
            model.output_shape
            model.summary()
            model.get_config()
            model.get_weights()

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=1)
            model.save(model_path)
        return model

    def train(self, x_train, y_train):
        model = self.get_model(x_train, y_train)
        return model

    def calc_additional(self, y_test, y_pred):
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion matrix: " + str(cm))

        # Precision
        ps = precision_score(y_test, y_pred)
        print("Precision: " + str(ps))

        # Recall
        rs = recall_score(y_test, y_pred)
        print("Recall: " + str(rs))

        # F1 score
        f1s = f1_score(y_test, y_pred)
        print("F1 score: " + str(f1s))

        # Cohen's kappa
        cks = cohen_kappa_score(y_test, y_pred)
        print("Cohen's kappa: " + str(cks))
