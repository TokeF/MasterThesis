import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.layers import Dense, Dropout, Flatten, Conv1D, regularizers
from keras.models import Sequential
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from NN.window import build_array_autoenc, build_vec
from utilities import data_visualize
from utilities.calc_metrics import metrics
from utilities.data_reader import load_data2, remove_edge
from utilities.data_visualize import plot_training, plot_misclassified


def cnn_autoenc(input_shape : tuple):
    model = Sequential()
    model.add(Conv1D(100, 6, activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))
    # model.add(BatchNormalization())
    model.add(Conv1D(100, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2()))
    # model.add(Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(MaxPooling1D(pool_size=3, strides=1, padding='same'))
    model.add(Flatten())
    # model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(700, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17*20, activation='linear'))

    return model

def simple_autoenc2(input_shape : tuple):
    model = Sequential()
    # model.add(Flatten(input_shape=input_shape))
    # model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    # model.add(Dropout(0.5))
    # model.add(Dense(340, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='sigmoid', input_shape=(17,)))
    # model.add(Dropout(0.5))
    model.add(Dense(5, activation='sigmoid'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='sigmoid'))
    # model.add(Dropout(0.5))
    model.add(Dense(17, activation='linear'))

    return model

def simple_autoenc(input_shape : tuple):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    # model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    # model.add(Dropout(0.5))
    # model.add(Dense(340, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(170, activation='relu'))
    model.add(Dense(85, activation='relu'))
    model.add(Dense(170, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(17*20, activation='linear'))

    return model

def correct_overlap_lbls(lbl_pred : np.ndarray, t_size : int, win : int, step : int):
    nanA = np.empty((lbl_pred.shape[0], t_size))
    nanA[:] = np.nan
    for i in range(0, lbl_pred.shape[0]):
        idx = i*step
        nanA[i, idx : idx + win] = lbl_pred[i]
    assert not np.isnan(nanA[-1, -1]), "last element did not get a score"

    mean_scor = np.nanmean(nanA, axis=0)
    lbl_cor = mean_scor < 0.1
    return lbl_cor.astype(int), mean_scor


def autoencoder2(X_train, X_val, lbl_val, X_valOG, timestamp_val, save = False):
    w = 20
    s = 3
    e = 100
    train_target = build_vec(X_train, w, s)
    X_train = build_array_autoenc(X_train, w, s)
    val_target = build_vec(X_val, w, s)
    X_val = build_array_autoenc(X_val, w, s)
    # train_target = X_train
    # test_target = X_test

    # 2. define autoencoder model
    model = simple_autoenc((w, X_train.shape[2]))
    # model = simple_autoenc2((w, X_train.shape[0]))
    print(model.summary())

    # 3. compile model
    model.compile(loss='mean_absolute_error',
          optimizer='Adam', metrics=['acc'])

    # 4. train model
    print("Starting training")
    # my_logger = MyLogger(n=100)
    h = model.fit(X_train, train_target, batch_size=64,
                  epochs=e, verbose=2, validation_data=(X_val, val_target))
    print("Training complete")

    # 6. Predict and correct labels
    predicteds = model.predict(X_val)
    mse = np.sum(np.square(predicteds - val_target), axis=1) / predicteds.shape[1]
    pred, mseCorrected = correct_overlap_lbls(mse, len(lbl_val), w, s)
    # mseCorrected = mse
    # pred = mse < 0.2
    hit = pred == lbl_val

    # 7. Metrics
    plot_misclassified(timestamp_val, X_valOG, lbl_val, pred)
    plot_training(h)
    report, CM, ACC, AUC = metrics(lbl_val, mseCorrected, pred)

    if save:
        model.save('auto_model.h5')  # creates a HDF5 file 'my_model.h5'
        # serialize model to JSON
        model_json = model.to_json()
        with open("auto_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("auto_weights.h5")
        print("Saved model to disk")

    return mseCorrected, report, ACC, AUC, CM


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
    timestamp=timestamp[0:r]

    splits = 5
    i = 0
    kf = KFold(n_splits=splits, shuffle=False)
    for trainidx, testidx in kf.split(dbdt):
        X_train, X_test = dbdt[trainidx], dbdt[testidx]
        y_train, y_test = lbl[trainidx], lbl[testidx]
        time_test = timestamp[testidx]
        X_testOG = X_test
        i+=1
        if i == 5: break

    X_train = X_train[y_train == 1, :]
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    fig, axROC = plt.subplots()
    ndata = np.linspace(0.01, 1, 50)
    AUCarray = np.zeros((len(ndata),))

    for i, frac in enumerate(ndata):
        n = int(np.ceil(frac * X_train.shape[0]))
        y_score, _, _ , _, CM = autoencoder2(X_train[0:n,:], X_test,
                                                y_test, X_testOG, time_test, True)
        AUC, _ = data_visualize.plot_roc(y_test, y_score, axROC, i, pos=0)
        AUCarray[i] = CM[1,0] + CM[0,1]
        plt.close('all')
        print(i)

    plt.figure("Learn data")
    plt.plot(ndata, AUCarray)
    plt.ylabel("Total errors")
    plt.xlabel("Training set in use [Fraction]")
    plt.show()
if __name__ == '__main__':
    main()

def autoencoder3(X_train, X_test, lbl_test):

    shape = X_train.shape[1]
    model = Sequential()
    model.add(Dense(15, activation='relu', input_dim=shape))
    model.add(Dropout(0.5))

    model.add(Dense(shape, activation='linear'))

    print(model.summary())
    # 3. compile model
    model.compile(loss='mean_squared_error',
          optimizer='Adam', metrics=['acc'])

    h = model.fit(X_train, X_train, batch_size=64,
                  epochs=9, verbose=2, validation_split=0.1)

    # 6. Predict and correct labels
    predicteds = model.predict(X_test)
    mse = np.sum(np.square(predicteds - X_test), axis=1) / predicteds.shape[1]
    lbl_pred = mse < 0.1
    hit = lbl_pred == lbl_test

    # 7. Metrics
    print(confusion_matrix(lbl_test, lbl_pred))
    report = classification_report(lbl_test, lbl_pred, output_dict=True)
    print(report)
    acc = sum(hit) / len(lbl_test)
    print("ACC", acc)
    # plot_misclassified(timestamp_test, X_testOG, lbl_test, lbl_pred)
    # plot_training(h)
    # plt.figure("ROC")
    # AUC = plot_roc(lbl_test, mseCorrected)
    # print("AUC ", AUC)

    return mse, report, acc