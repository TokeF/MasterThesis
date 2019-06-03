import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from NN.window import build_array_skytem
# from neupy.layers import *
# from neupy import algorithms


## Data preprocessing
from utilities.data_reader import load_data2, remove_edge
from utilities.data_visualize import plot_training, plot_misclassified, dist_score

fname2 = "../data/stendalmark_20181122_RAW_export.xyz"
fname1 = "../data/stendalmark_20181121_RAW_export.xyz"
fname0 = "../data/stendalmark_20181120_RAW_export.xyz"
fname = "../data/vildbjerg_20171101_RAW_export.xyz"
df, dbdt, lbl, timestamp, _ = load_data2(fname1, 8, 24)
dbdt, lbl, timestamp = remove_edge(timestamp, dbdt, lbl, 20)

if 1:
    #normalize by sign and log
    dbdt_norm = np.sign(dbdt) * np.log(np.abs(dbdt))
    #split in rtaining test
    r = int(np.ceil(0.8 * dbdt.shape[0]))
    dbdt_train = dbdt_norm[0:r, :]
    lbl_train = lbl[1:r-1]
    dbdt_testOG = dbdt[r+1:-1]
    dbdt_test = dbdt_norm[r:]
    lbl_test = lbl[r+1:-1]
    #normalise by zero mean, 1 std
    sc = StandardScaler()
    dbdt_train = sc.fit_transform(dbdt_train)
    dbdt_test = sc.transform(dbdt_test)
    #Build input array, where a sounding and its neighbours are stacked
    dbdt_train = build_array_skytem(3, dbdt_train)
    dbdt_test = build_array_skytem(3, dbdt_test)
else:
    from sklearn.model_selection import train_test_split
    #normalize by sign and log
    dbdt_norm = np.sign(dbdt) * np.log(np.abs(dbdt))
    sc = StandardScaler()
    dbdt_norm = sc.fit_transform(dbdt_norm)
    dbdt_norm = build_array_skytem(3, dbdt_norm)
    X = range(dbdt_norm.shape[1])
    X_train_idx, X_test_idx, lbl_train, lbl_test = train_test_split(X, lbl[1:-1], test_size=0.2)
    dbdt_train = dbdt_norm[:, X_train_idx]
    dbdt_test = dbdt_norm[:, X_test_idx]

model = Sequential()
model.add(Dense(30, input_shape=(dbdt_train.shape[0],), activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='binary_crossentropy',
              optimizer='Adam',#Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])
print(model.summary())
history = model.fit(dbdt_train.T, lbl_train, epochs=30, batch_size = 100,
                    verbose = 2, validation_split=0.1)

# network = join(
#     Input(51),
#     Tanh(30),
#     Tanh(1),
# )
#
# model2 = algorithms.LevenbergMarquardt(network)
# model2.train(dbdt_train, lbl_train)

## compute metrics
lbl_pred_scor = model.predict(dbdt_test.T)
lbl_pred = lbl_pred_scor > 0.63
plt.figure("score")
dist_score(lbl_pred_scor)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(lbl_test, lbl_pred))
print(classification_report(lbl_test, lbl_pred))
hit = lbl_pred[0] == lbl_test
print("score: " + str(sum(lbl_pred[0] == lbl_test) / len(lbl_test)))

plot_training(history)
# plot_misclassified(timestamp[r+1:-1], dbdt_testOG, lbl_test, lbl_pred)

plt.show()
