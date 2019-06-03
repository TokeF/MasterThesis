from __future__ import print_function
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D


def cnn1D(input_shape : tuple):
    model = Sequential()
    model.add(Conv1D(100, 6, activation='relu', padding='same', input_shape=input_shape))#, kernel_regularizer=regularizers.l2(0.01)))
    # model.add(BatchNormalization())
    model.add(Conv1D(100, 3, activation='relu', padding='same'))#, kernel_regularizer=regularizers.l2()))
    # model.add(Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01)))
    # model.add(MaxPooling1D(pool_size=3, strides=1, padding='same'))
    model.add(Flatten())
    # model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

