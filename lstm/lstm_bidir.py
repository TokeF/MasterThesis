import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras.backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import LSTM, TimeDistributed, Dense, Dropout, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from utilities.data_reader import load_data2
from utilities.data_visualize import plot_training
from NN.window import build_array_lstm

def weighted_binary_crossentropy(y_true, y_pred) :
    logloss = -(y_true * K.log(y_pred) * 0.25 + (1 - y_true) * K.log(1 - y_pred) * 0.75)
    return K.mean(logloss, axis=-1)


def lstm_bidir(dbdt_train : np.ndarray, lbl_train : np.ndarray,
               dbdt_test : np.ndarray, lbl_test : np.ndarray, stepsize : int):
    print("dbdt train: " + str(dbdt_train.shape))
    print("lbl train: " + str(lbl_train.shape))
    print("dbdt test: " + str(dbdt_test.shape))
    print("lbl test: " + str(lbl_test.shape))

    model = Sequential()
    model.add(Bidirectional(
        LSTM(10, batch_input_shape=(dbdt_train.shape[0], dbdt_train.shape[1], dbdt_train.shape[2])
               , return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(20, return_sequences=True)))
    # model.add(TimeDistributed(Dense(1, activation='tanh')))
    model.add(Dropout(0.4))
    model.add(TimeDistributed(Dense(10, activation='sigmoid')))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(loss=weighted_binary_crossentropy,
                  optimizer=Adam(lr=0.00324, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['acc'])

    history = model.fit(dbdt_train, lbl_train, epochs=100, batch_size=64,
                        verbose=2, validation_split=0.1, shuffle=False)

    print(model.summary())

    # scores are predicted between 0 and 1 for each window. Correct scores for overlapping windows
    lbl_pred_scor = model.predict(dbdt_test)
    lbl_pred_scor = correct_overlap_lbls(lbl_pred_scor, stepsize)
    plt.figure()
    sns.distplot(lbl_pred_scor)
    lbl_pred = lbl_pred_scor > 0.85
    lbl_pred.astype(int)

    print("score: " + str(sum(lbl_pred) / len(lbl_test)))
    print(confusion_matrix(lbl_test, lbl_pred))
    print(classification_report(lbl_test, lbl_pred))

    plot_training(history)
    plt.show()


def correct_overlap_lbls(lblA : np.ndarray, stepzise : int):
    # Length og original data: sequence length, plus stepsize for each row other than the first
    # ie. the total number og soundings, accounting for overlapping windows
    nsounding = lblA.shape[1] + (lblA.shape[0] - 1) * stepzise
    # Create empty array. Each row holds predictions for the corresponding window columns.
    # remaining columns are NaN. Average the score of a sounding over each windows prediction
    nanA = np.empty((lblA.shape[0], nsounding))
    nanA[:] = np.nan
    for i in range(0, lblA.shape[0]):
        idx = i*stepzise
        nanA[i, idx : idx + lblA.shape[1]] = lblA[i, :, 0]
    assert np.isnan(nanA[-1, -1]), "last element did not get a score"
    return np.nanmean(nanA, axis=0)


fname = "../data/stendalmark_20181120_RAW_export.xyz"
df, dbdt, lbl, timestamp, gtimes = load_data2(fname, 8, 24)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

r = int(np.ceil(0.8 * dbdt.shape[0]))
X_train = dbdt[0:r, :]
X_test = dbdt[r:, :]
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

lbl_train2 = lbl[0:r]
lbl_test = lbl[r:]



w = 30
s = 1
X_train, lbl_train = build_array_lstm(X_train, w, s, lbl_train2)
X_test, _ = build_array_lstm(X_test, w, s, lbl_test)
X_train = X_train[0:-1, :, :]
X_test = X_test[0:-1, :, :]
lbl_train = lbl_train[0:-1, :, :]


nsounding = X_test.shape[1] + (X_test.shape[0] - 1) * s
lbl_test = lbl_test[0:nsounding]

lstm_bidir(X_train, lbl_train, X_test, lbl_test, s)


## used when trying to overfit
# X_test = X_train

# nsounding = X_train.shape[1] + (X_train.shape[0] - 1) * s
# lbl_test = lbl_train2[0:nsounding]