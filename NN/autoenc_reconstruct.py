from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from utilities import data_visualize
from utilities.data_reader import load_data2, remove_edge

# Load data and do windowing
fname0 = "../data/stendalmark_20181120_RAW_export.xyz"
fname1 = "../data/stendalmark_20181121_RAW_export.xyz"

df, dbdt, lbl, timestamp, gtimes = load_data2(fname0, 8, 24)
dbdt, lbl, timestamp = remove_edge(timestamp, dbdt, lbl, 20)
timestamp = (timestamp - timestamp[0]) * 10 ** 5

df0, dbdt0, lbl0, timestamp0, gtimes0 = load_data2(fname1, 8, 24)
dbdt0, lbl0, timestamp0 = remove_edge(timestamp0, dbdt0, lbl0, 20)
timestamp0 = (timestamp0 - timestamp0[0]) * 10 ** 5 + timestamp[-1]+0.7

dbdt = np.concatenate((dbdt,dbdt0)); lbl = np.concatenate((lbl,lbl0))
timestamp = np.concatenate((timestamp, timestamp0))

# idx = range(17368, 17368+20*4)
idx = range(28560, 28660)
window = dbdt[idx,:]
lbl_window = lbl[idx]
time_window = timestamp[idx]


data_visualize.plotDat(time_window, window, lbl_window)
plt.yscale("log")
plt.ylim([10**(-10), 10**(-5)])
trainidx = int(np.ceil(0.8*dbdt.shape[0]))
sc = StandardScaler()
X_train = sc.fit(dbdt[0:trainidx])
window = sc.transform(window)
model_mem = load_model("auto_model.h5")
model_mem.compile(loss='mean_squared_error',
          optimizer='Adam', metrics=['acc'])
print(model_mem.summary())
window = window.reshape(5, 20, 17)
score_mem = model_mem.predict(window)
score_mem = score_mem.reshape((5*20, 17))

# Denormalize datapoints
score_mem = sc.inverse_transform(score_mem)

plt.figure(4)
data_visualize.plotDat(time_window, score_mem, lbl_window)
# plt.plot(time_window, score_mem, 'o-')
plt.yscale("log")
plt.ylim([10**(-10), 10**(-5)])
plt.show()
