import sys
sys.path.append("..")
from utilities.data_visualize import plot_misclassified
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from NN.window import build_array_skytem
from neupy.layers import *
from neupy import algorithms


## Data preprocessing
from utilities.data_reader import load_data2, remove_edge

# from utilities.data_visualize import plot_training, plot_misclassified

fname2 = "../data/stendalmark_20181122_RAW_export.xyz"
fname1 = "../data/stendalmark_20181121_RAW_export.xyz"
fname0 = "../data/stendalmark_20181120_RAW_export.xyz"
fname = "../data/vildbjerg_20171101_RAW_export.xyz"
df, dbdt, lbl, timestamp, _ = load_data2(fname1, 8, 24)
dbdt, lbl, timestamp = remove_edge(timestamp, dbdt, lbl, 20)
#normalize by sign and log
dbdt_norm = np.sign(dbdt) * np.log(np.abs(dbdt))
#split in rtaining test
r = int(np.ceil(0.8 * dbdt.shape[0]))
dbdt_train = dbdt_norm[0:r, :]
lbl_train = lbl[1:r-1]
# lbl_train[lbl_train==0] = -1
dbdt_testOG = dbdt[r+1:-1]
dbdt_test = dbdt_norm[r:]
lbl_test = lbl[r+1:-1]
# lbl_test[lbl_test==0] = -1
#normalise by zero mean, 1 std
sc = StandardScaler()
dbdt_train = sc.fit_transform(dbdt_train)
dbdt_test = sc.transform(dbdt_test)
#Build input array, where a sounding and its neighbours are stacked
dbdt_train = build_array_skytem(3, dbdt_train)
dbdt_test = build_array_skytem(3, dbdt_test)


network = join(
    Input(51),
    Tanh(30),
    Tanh(1),
)

model2 = algorithms.LevenbergMarquardt(network, verbose=True)
model2.train(dbdt_train.T, lbl_train.T, epochs=10)

# ## compute metrics
lbl_pred_scor = model2.predict(dbdt_test.T)
lbl_pred = lbl_pred_scor > 0
print(lbl_pred)
print(sum(lbl_pred[0]==lbl_test) / len(lbl_test))

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(lbl_test, lbl_pred))
print(classification_report(lbl_test, lbl_pred))

#
# plot_training(history)
# plot_misclassified(timestamp[r+1:-1], dbdt_testOG, lbl_test, lbl_pred)
# plt.show()
