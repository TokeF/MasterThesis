import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from keras.activations import softmax
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
# numpy.random.seed(7)
import utilities.difference
from utilities.data_reader import load_data2
from sklearn.model_selection import train_test_split
# # load the dataset but only keep the top n words, zero the rest
# fname = "../data/20171101_RAW_export.xyz"
# _ , dbdt, lbl, timestamp = load_data2(fname, 8, 20)

def softMaxAxis1(x):
    return softmax(x,axis=1)

def lstm_mini(dbdt, lbl):
    # dbdt = difference.row_ratio(timestamp, dbdt)
    # X_train, X_test, lbl_train, lbl_test = train_test_split(dbdt, lbl, test_size = 0.30)
    r = int(numpy.ceil(0.8 * dbdt.shape[0]))
    X_train = dbdt[0:r, :]
    X_test = dbdt[r:, :]
    lbl_train = lbl[0:r]
    lbl_test= lbl[r:]

    #feature scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler()
    X_train, lbl_train = rus.fit_sample(X_train, lbl_train)

    X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    # create the model
    model = Sequential()
    # model.add(Dense(32, input_shape=(1,X_train.shape[2])))
    # model.add(Dropout(0.2))
    model.add(LSTM(100, input_shape=(1, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, lbl_train, epochs=1000, batch_size = 64, verbose = 1)
    # Final evaluation of the model
    scores = model.evaluate(X_test, lbl_test, verbose=0)
    lbl_pred_scor = model.predict(X_test)
    lbl_pred = lbl_pred_scor.round()
    print(lbl_pred)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(lbl_test,lbl_pred))
    print(classification_report(lbl_test,lbl_pred))