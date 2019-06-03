import numpy as np
import json
from matplotlib import pyplot as plt
# np.random.seed(123)  # for reproducibility
#import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import plot_model
#utilities
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import load_model
# from livelossplot.keras import PlotLossesCallback
#import data
from keras.datasets import mnist
 
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
plt.imshow(X_train[0])

#reshape and scale data to fit network input
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
print(X_train.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

#define model and layers
model = Sequential()
 
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit model
if True:
    # plot_losses = PlotLossesCallback()
    #fit model to data
    history = model.fit(X_train, Y_train, 
                        batch_size=32, 
                        epochs=5,
                        # callbacks=[plot_losses],
                        verbose=1)
    with open('histFile.json', 'w') as f:
        json.dump(history.history, f)

    #evaluate model
    score = model.evaluate(X_test, Y_test, verbose=0)

    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    json_string = model.to_json()
    # with open("model.json", "w") as json_file:
    #     # json_file.write(model_json)
    del model  # deletes the existing model

#load model
if False:
    model_mem = load_model("my_model.h5")
    score_mem = model_mem.evaluate(X_test, Y_test, verbose=0)
    print(type(score_mem))
    print(score_mem)