from __future__ import print_function
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras.backend as K
from keras.optimizers import Adam
from sklearn.model_selection import KFold

from utilities import data_visualize
from utilities.calc_metrics import metrics
from utilities.data_reader import load_data2, remove_edge
from NN.window import build_array
from NN.cnn1D import cnn1D
from utilities.data_visualize import plot_training, plot_misclassified
from sklearn.preprocessing import StandardScaler


def weighted_binary_crossentropy(y_true, y_pred) :
    logloss = -(y_true * K.log(y_pred) * 1 +
                (1 - y_true) * K.log(1 - y_pred) * 4)
    return logloss


def correct_overlap_lbls(lbl_pred : np.ndarray, t_size : int, win : int, step : int):
    nanA = np.empty((lbl_pred.shape[0], t_size))
    nanA[:] = np.nan
    for i in range(0, lbl_pred.shape[0]):
        idx = i*step
        nanA[i, idx : idx + win] = lbl_pred[i]
    assert not np.isnan(nanA[-1, -1]), "last element did not get a score"

    mean_scor = np.nanmean(nanA, axis=0)
    lbl_cor = mean_scor > 0.87
    # plt.figure()
    # sns.distplot(mean_scor)
    return lbl_cor.astype(int), mean_scor

def runCNN(timestamp, X_train, X_val, lbl_train, lbl_val,
           X_testOG, timestamp_test, build = True,):
    w = 20
    s = 3
    if build:
        X_train, lbl_train = build_array(X_train, w, s, lbl_train)
        X_val, lbl_valA = build_array(X_val, w, s, lbl_val)
    print("Uncoupled: " + str(sum(lbl_train) / len(lbl_train)))


    # Build CNN model
    print("X train: " + str(X_train.shape))
    print("y train: " + str(lbl_train.shape))
    print("X test: " + str(X_val.shape))
    print("y test: " + str(lbl_val.shape))
    model = cnn1D((w, X_train.shape[2]))
    # model = resnetmodel()

    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',#Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['accuracy'])
    print(model.summary())

    # lbl_train = to_categorical(lbl_train, 2)
    history = model.fit(X_train, lbl_train, epochs=64, batch_size = 1000, verbose = 2,
                        validation_data =(X_val, lbl_valA))

    ##
    lbl_pred_scor = model.predict(X_val)
    lbl_pred, y_score = correct_overlap_lbls(lbl_pred_scor, len(lbl_val), w, s)

    report, CM, ACC, AUC = metrics(lbl_val, y_score, lbl_pred)

    plot_training(history)
    plot_misclassified(timestamp_test, X_testOG, lbl_val, lbl_pred)

    model.save('model2.h5')  # creates a HDF5 file 'my_model.h5'
    # serialize model to JSON
    model_json = model.to_json()
    with open("model2.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model2.h5")
    print("Saved model to disk")
    return y_score, report, ACC, AUC, CM


def main():
    fname0 = "../data/stendalmark_20181120_RAW_export.xyz"
    fname1 = "../data/stendalmark_20181121_RAW_export.xyz"

    df, dbdt, lbl, timestamp, gtimes = load_data2(fname0, 8, 24)
    dbdt, lbl, timestamp = remove_edge(timestamp, dbdt, lbl, 20)
    timestamp = (timestamp - timestamp[0]) * 10 ** 5

    df0, dbdt0, lbl0, timestamp0, gtimes0 = load_data2(fname1, 8, 24)
    dbdt0, lbl0, timestamp0 = remove_edge(timestamp0, dbdt0, lbl0, 20)
    timestamp0 = (timestamp0 - timestamp0[0]) * 10 ** 5 + timestamp[-1] + 0.7

    dbdt = np.concatenate((dbdt, dbdt0));
    lbl = np.concatenate((lbl, lbl0))
    timestamp = np.concatenate((timestamp, timestamp0))
    sc = StandardScaler()

    r = int(np.ceil(0.8 * dbdt.shape[0]))
    dbdt = dbdt[0:r, :]
    lbl = lbl[0:r]
    timestamp = timestamp[0:r]

    splits = 5
    i = 0
    kf = KFold(n_splits=splits, shuffle=False)
    for trainidx, testidx in kf.split(dbdt):
        X_train, X_test = dbdt[trainidx], dbdt[testidx]
        y_train, y_test = lbl[trainidx], lbl[testidx]
        time_test = timestamp[testidx]
        X_testOG = X_test
        i += 1
        if i == 5: break


    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    fig, axROC = plt.subplots()
    ndata = np.linspace(0.01, 1, 50)
    AUCarray = np.zeros((len(ndata),))

    for i, frac in enumerate(ndata):
        n = int(np.ceil(frac * X_train.shape[0]))
        y_score, report, ACC, AUC, CM = runCNN(timestamp,X_train[0:n,:], X_test, y_train[0:n], y_test,
           X_testOG, time_test, build = True,)
        AUC, _ = data_visualize.plot_roc(y_test, y_score, axROC, i, pos=0)
        AUCarray[i] = CM[1, 0] + CM[0, 1]
        plt.close('all')
        print(i)

    plt.figure("Learn data")
    plt.plot(ndata, AUCarray)
    plt.ylabel("Total errors")
    plt.xlabel("Training set in use [Fraction]")
    plt.show()

if __name__ == '__main__':
    main()
