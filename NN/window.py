import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, Flatten
# from utilities.data_reader import load_data2
# from utilities.data_visualize import plotDat


def sliding_window(dbdt : np.ndarray, stepsize : int, windowsize : int, lbl : np.ndarray):
    # slide a window across the image
    for x in range(0, dbdt.shape[0] - windowsize, stepsize):
        # yield the current window
        coupled = lbl[x:x + windowsize] == 0
        yield (x, dbdt[x:x + windowsize, :], coupled)


def build_array(dbdt : np.ndarray, windowsize : int, stepsize : int, lbl : np.ndarray):
    # initialize matrix to keep the windowed data. Last dimension is how many steps we can take.
    nwindows = int(np.ceil((dbdt.shape[0] - windowsize) / stepsize) + 1)
    tens = np.zeros((nwindows, windowsize, dbdt.shape[1]))
    y = np.zeros(nwindows)
    # slide a window across the image
    for i, x in enumerate(range(0, dbdt.shape[0] - (windowsize-stepsize), stepsize)):
        # find which label 0 or 1, occur most often in the window. Use this as image label
        y[i] = lbl[x + int(windowsize/2)] #int(np.round(sum(lbl[x:x + windowsize]) / len(lbl[x:x + windowsize])))
        if i == nwindows - 1:
            wend = x + windowsize + 1
            dbdtend = dbdt[x:wend, :]
            for h in range(0, dbdtend.shape[0]):
                tens[i, h, :] = dbdtend[h,:]
                print(h)
        else:
            wend = x + windowsize
            tens[i, :, :] = dbdt[x:wend, :]
    return tens, y

def build_vec(dbdt : np.ndarray, windowsize : int, stepsize : int):
    # initialize matrix to keep the windowed data. Last dimension is how many steps we can take.
    tens = np.zeros((int(np.ceil(dbdt.shape[0] / stepsize)), windowsize * dbdt.shape[1]))
    # slide a window across the image
    for i, x in enumerate(range(0, dbdt.shape[0] - windowsize + 1, stepsize)):
        tens[i, :] = dbdt[x:x + windowsize, :].flatten()
    return tens

def build_array_lstm(dbdt : np.ndarray, windowsize : int, stepsize : int, lbl : np.ndarray):
    samples = np.zeros((int(np.ceil((dbdt.shape[0] - windowsize) / stepsize)), windowsize, dbdt.shape[1]))
    labels = np.zeros((int(np.ceil((dbdt.shape[0] - windowsize) / stepsize)), windowsize))
    for i, x in enumerate(range(0, dbdt.shape[0] - windowsize, stepsize)):
        samples[i, :, :] = dbdt[x:x + windowsize, :]
        labels[i, :] = lbl[x:x + windowsize]
    labels = np.reshape(labels, (labels.shape[0], labels.shape[1], 1))
    return samples, labels

def build_array_skytem(hood:int, dbdt:np.ndarray):
    #Create NaN tensor of size 3*gates, to hold sounding plus two neighbours
    #second dimension is all soudnings minus first and last, since they dont have neighbor
    tens = np.empty((dbdt.shape[1] * hood, dbdt.shape[0] - (hood-1)))
    tens[:] = np.nan
    for i in range(0, dbdt.shape[0] - (hood-1)):
        tens[:,i] = np.reshape(dbdt[i : i+hood, :], (dbdt.shape[1]*hood))
    assert not np.isnan(tens).any(), "NaN value in input tensor"
    return tens

def build_array_autoenc(dbdt : np.ndarray, windowsize : int, stepsize : int):
    # initialize matrix to keep the windowed data. Last dimension is how many steps we can take.
    tens = np.zeros((int(np.ceil(dbdt.shape[0] / stepsize)), windowsize, dbdt.shape[1]))
    # slide a window across the image
    for i, x in enumerate(range(0, dbdt.shape[0] - windowsize + 1, stepsize)):
        tens[i, :, :] = dbdt[x:x + windowsize, :]
    return tens
#
# def driver_thing():
#     fname = "../data/vildbjerg_20171101_RAW_export.xyz"
#     # fname = "../data/stendalmark_20181120_RAW_export.xyz"
#     df, dbdt, lbl, timestamp, gtimes = load_data2(fname, 8, 24)
#
#     # dbdt = np.log10(np.abs(dbdt))
#     # normalized = (dbdt-np.amin(dbdt))/(np.amax(dbdt)-np.amin(dbdt))
#     from sklearn.preprocessing import StandardScaler, MinMaxScaler
#     mm = MinMaxScaler()
#     sc = StandardScaler()
#     dbdt = mm.fit_transform(dbdt)
#     # dbdt = sc.fit_transform(dbdt)
#     tens, y = build_array(dbdt, 100, 100, lbl)
#
#     print(tens.shape)
#     print(y.shape)
#     # plt.imshow(tens[:, :, -2].T)
#     # plt.show()
#
#     X = tens.reshape(tens.shape[2], tens.shape[1], tens.shape[0], 1)
#     # y = y.reshape(y.shape[0], y.shape[1], y.shape[0], 1)
#     # idx = list(range(tens.shape[2]))
#     # np.random.shuffle(idx)
#     r = int(np.ceil(0.8 * X.shape[0]))
#     X_train = X[0:r, :, :, :]
#     X_test = X[r:-1, :, :, :]
#     y_train = y[0:r]
#     y_test = y[r:-1]
#     # y_lbl = y_test
#
#     from keras.utils import to_categorical
#     y_train = to_categorical(y_train)
#     y_test = to_categorical(y_test)
#     print(y_train.shape)
#     print(X_train.shape)
#     print(y_test.shape)
#     print(X_test.shape)
#
#
#     for i in range(X_test.shape[0] - 1):
#         img = X_test[i, :, :, 0]
#         plt.figure()
#         plt.plot(range(1,101), img.T)
#         plt.figure()
#         img2, _ = image_histogram_equalization(img)
#         plt.imshow(img2)
#         plt.show()
#
#     exit()
#     #create model
#     model = Sequential()
#     #add model layers
#     model.add(Conv2D(100, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
#     model.add(Flatten())
#     model.add(Dense(2, activation='softmax'))
#
#     #compile model using accuracy as a measure of model performance
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
#     #train model
#     model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=5)
#
#     # metric
#     y_scor = model.predict(X_test)
#     y_pred = np.argmax(y_scor, axis=1)
#     y_test = np.argmax(y_test, axis=1)
#     from sklearn.metrics import classification_report, confusion_matrix
#     print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
#
#
#
#
#     myGenerator = sliding_window(X_test, 30, 30, lbl)
#
#     # for x, w, coupled in myGenerator:
#     #     plt.imshow(w)
#     #     mark = np.where(coupled)[0]
#     #     for m in mark:
#     #         plt.axvline(x=m, color='red')
#     #     plt.show()
#
#
#
# def image_histogram_equalization(image, number_bins=256):
#     # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html
#
#     # get image histogram
#     image_histogram, bins = np.histogram(image.flatten(), number_bins)
#     cdf = image_histogram.cumsum() # cumulative distribution function
#     cdf = 255 * cdf / cdf[-1] # normalize
#
#     # use linear interpolation of cdf to find new pixel values
#     image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
#
#     return image_equalized.reshape(image.shape), cdf
#
#
# # driver_thing()
