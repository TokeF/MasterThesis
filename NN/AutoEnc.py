import numpy as np
# import keras as K
from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from NN.window import build_array_skytem, build_array_autoenc, build_vec
from utilities.data_reader import remove_edge, load_data2, timestampToTime
from utilities.data_visualize import plot_training, plot_misclassified, plot_roc
import datetime

def cus_accuracy(y_true, y_pred):
    N = y_pred._keras_shape
    mse = K.sum(K.square(y_true - y_pred), axis=1) / N[1]
    return mse

def correct_overlap_lbls(lbl_pred : np.ndarray, t_size : int, win : int, step : int):
    nanA = np.empty((lbl_pred.shape[0], t_size))
    nanA[:] = np.nan
    for i in range(0, lbl_pred.shape[0]):
        idx = i*step
        nanA[i, idx : idx + win] = lbl_pred[i]
    assert not np.isnan(nanA[-1, -1]), "last element did not get a score"

    mean_scor = np.nanmean(nanA, axis=0)
    lbl_cor = mean_scor < 10
    plt.figure()
    sns.distplot(mean_scor,700)
    return lbl_cor.astype(int), mean_scor

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
    model.add(Dense(340, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(170, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17*20, activation='linear'))

    return model

def simple_autoenc(input_shape : tuple):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    # model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(340, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(170, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(85, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(170, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17*20, activation='linear'))

    return model

def train_fit(model, X_train, train_target, X_test, test_target, lbl_test, epochs, w, s):
    # 4. train model
    print("Starting training")
    # my_logger = MyLogger(n=100)
    h = model.fit(X_train, train_target, batch_size=64,
                  epochs=epochs, verbose=2, validation_split=0.1)
    print("Training complete")

    # 5. TO DO: save model
    # 6. metrics

    predicteds = model.predict(X_test)
    mse = np.sum(np.square(predicteds - test_target), axis=1) / predicteds.shape[1]
    # sns.distplot(mse)
    # plt.xscale('log')

    pred, mseCorrected = correct_overlap_lbls(mse, len(lbl_test), w, s)
    hit = pred == lbl_test
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(lbl_test, pred))
    report = classification_report(lbl_test, pred, output_dict=True)
    print(report)
    acc = sum(hit) / len(lbl_test)
    print("ACC", acc)
    # plot_misclassified(timestamp[r:], dbdt_testOG, lbl_test, pred)
    plot_training(h)
    plt.figure("ROC")
    AUC = plot_roc(lbl_test, mseCorrected)
    print("AUC ", AUC)

    return mseCorrected, report, acc

def autoencoder(dbdt_train, dbdt_test, lbl_test):
    w = 20
    s = 3
    X_train = build_array_autoenc(dbdt_train, w, s)
    train_target = build_vec(dbdt_train, w, s)
    X_test = build_array_autoenc(dbdt_test, w, s)
    test_target = build_vec(dbdt_test, w, s)

    # 2. define autoencoder model
    autoenc = simple_autoenc((w, X_train.shape[2]))
    print(autoenc.summary())

    # 3. compile model
    autoenc.compile(loss='mean_squared_error',
          optimizer='Adam', metrics=['acc'])

    y_score, report, acc = train_fit(autoenc, X_train, train_target, X_test, test_target, lbl_test, 40, w, s)

    return y_score, report, acc


def main():
    # 1. load data
    fname2 = "../data/stendalmark_20181122_RAW_export.xyz"
    fname1 = "../data/stendalmark_20181121_RAW_export.xyz"
    fname0 = "../data/stendalmark_20181120_RAW_export.xyz"
    fname = "../data/vildbjerg_20171101_RAW_export.xyz"
    _, dbdt, lbl, timestamp, _ = load_data2(fname1, 8, 24)
    _, dbdt1, lbl1, timestamp1, _ = load_data2(fname0, 8, 24)
    _, dbdt2, lbl2, timestamp2, _ = load_data2(fname, 8, 24)
    dbdt, lbl, timestamp = remove_edge(timestamp, dbdt, lbl, 20)
    dbdt1, lbl1, timestamp1 = remove_edge(timestamp1, dbdt1, lbl1, 20)
    dbdt2, lbl2, timestamp2 = remove_edge(timestamp2, dbdt2, lbl2, 20)

    timestamp = timestampToTime(timestamp)

    # normalize by sign and log
    # dbdt_norm = np.sign(dbdt) * np.log(np.abs(dbdt))
    # split in training test
    r = int(np.ceil(0.8 * dbdt.shape[0]))
    dbdt_train = dbdt[0:r, :]
    lbl_train = lbl[0:r]
    dbdt_train = dbdt_train[lbl_train == 1, :]
    # dbdt_train = np.concatenate(( dbdt1[lbl1 == 1, :], dbdt2[lbl2 == 1, :]), axis=0)

    dbdt_testOG = dbdt[r:, :]
    dbdt_test = dbdt[r:, :]
    lbl_test = lbl[r:]
    # normalise by zero mean, 1 std
    sc = StandardScaler()
    dbdt_train = sc.fit_transform(dbdt_train)
    dbdt_test = sc.transform(dbdt_test)
    # Build input array, where a sounding and its neighbours are stacked
    autoencoder(dbdt_train, dbdt_test, lbl_test)


if __name__ == '__main__':
    main()

