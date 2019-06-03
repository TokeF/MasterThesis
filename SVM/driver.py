import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
import sys

from sklearn.preprocessing import StandardScaler

sys.path.append("..")  # Adds higher directory to python modules path.
from utilities import data_visualize, difference, PC_anal
from utilities.data_reader import load_data2, remove_edge, timestampToTime
from SVM.rndForest import rnd_forest
from SVM.rndForest2 import rnd_forest2 as rf2
from lstm.lstm_one import lstm_mini
# from SVM.SVM_1 import SVM_classify

# fig, ax = plt.subplots()
# ax.plot([0, 0, 1], [0, 1, 1], "r-", label="Perfect classifier")
# ax.set_ylim([0,1])
# ax.set_xlim([0,1])
# ax.plot([0,1], [0,1], color="tab:gray", linestyle = "-.", label="50% line")
# ax.set_ylabel('True positive rate'); ax.set_xlabel('False positive rate')
# ax.legend()
# plt.show()
# exit()
fname = "../data/vildbjerg_20171101_modified_RAW_export.xyz"
fname1 = "../data/stendalmark_20181121_RAW_export.xyz"
fname0 = "../data/stendalmark_20181120_RAW_export.xyz"

df, dbdt, lbl, timestamp, gtimes = load_data2(fname0, 8, 24)
dbdt, lbl, timestamp = remove_edge(timestamp, dbdt, lbl, 20)
timestamp = (timestamp - timestamp[0]) * 10 ** 5

df0, dbdt0, lbl0, timestamp0, gtimes0 = load_data2(fname1, 8, 24)
dbdt0, lbl0, timestamp0 = remove_edge(timestamp0, dbdt0, lbl0, 20)
timestamp0 = (timestamp0 - timestamp0[0]) * 10 ** 5 + timestamp[-1]+0.7

dbdt = np.concatenate((dbdt,dbdt0)); lbl = np.concatenate((lbl,lbl0))
timestamp = np.concatenate((timestamp, timestamp0))

print("coupled ",sum(lbl == 0) / dbdt.shape[0])
print("total coupled ",dbdt[lbl==0,:].shape[0])
print("total", dbdt.shape[0])

data_visualize.plotDat(range(0, len(timestamp)), dbdt, lbl)
plt.yscale('log')
plt.figure()
plt.plot(range(0,len(lbl)), lbl, '.')
plt.show()
exit()

scor, frank,_, _ = rf2(timestamp, dbdt, lbl, n_trees=100)

# folds = 10
# metrics = np.zeros([8,folds])
# for i in range(folds):
#     _,_, report, acc = rnd_forest(timestamp, dbdt, lbl, n_trees=10)
#     metrics[0,i] = report.get("0").get("precision")
#     metrics[1, i] = report.get("0").get("recall")
#     metrics[2, i] = report.get("1").get("precision")
#     metrics[3, i] = report.get("1").get("recall")
#     metrics[4, i] = report.get("0").get("support")
#     metrics[5, i] = report.get("1").get("support")
#     metrics[6, i] = report.get("weighted avg").get("f1-score")
#     metrics[7, i] = acc
# print("mean acc: " + str(metrics[7,:].mean()) + " +- " + str(metrics[7,:].std()))
plt.figure("Score distribution")
# plt.bar(range(1,18), frank)
data_visualize.dist_score(scor[:,1])
plt.show()
exit()

# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# sc = StandardScaler()
# mm = MinMaxScaler()
# # dbdt = mm.fit_transform(dbdt)
# dbdt = sc.fit_transform(dbdt)
# lstm_mini(dbdt, lbl)
# data_visualize.plotDat(timestamp, dbdt, lbl)
# plt.yscale('log')
# plt.show()
# exit()
# plt.plot(gtimes[0, :], dbdt[100, :].T)
# plt.yscale('log')
# plt.xscale('log')
# plt.show()

# data_visualize.histograms(dbdt, lbl)
# dbdt = StandardScaler().fit_transform(dbdt)
dbdt = np.sign(dbdt) * np.log(np.abs(dbdt))
data_visualize.plotDat(timestamp, dbdt, lbl)
plt.yscale('log')
plt.show()
exit()

#
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# dbdt = sc.fit_transform(dbdt)

# data_visualize.plotDat(timestamp, dbdt, lbl)
# plt.xlim((0.0014 + 4.30404e4, 0.0028 + 4.30404e4))
# plt.yscale('log')

lr, rl = difference.mvg_var_leftright(pd.DataFrame(dbdt), 10)
lr = lr[10:-10]
rl = rl[10:-10]
lbl = lbl[10:-10]
timestamp = timestamp[10:-10]
# r = np.abs(lr - rl)
# r = np.sqrt(rl**2 + lr**2)
# r = lr / rl
r = np.maximum(lr, rl)

# data_visualize.histograms(r, lbl, 500, kdebool=False, normhist=True)
# plt.yscale('log')
# plt.xlim((0.0014 + 4.30404e4, 0.0028 + 4.30404e4))
# plt.show()

supers = np.concatenate((dbdt[10:-10], r), axis=1)
rnd_forest(timestamp, supers, lbl, n_trees=10000)

exit()
slope, rvalue = difference.linear_deriv(timestamp, dbdt, 11)
rvalue = np.square(rvalue)

# indicesToKeep = lbl == 1
# uncoupled = rvalue[indicesToKeep, :];
# uncoupled = np.reshape(uncoupled, (uncoupled.shape[0] * uncoupled.shape[1], ))
# coupled = rvalue[~indicesToKeep, :]
# coupled = np.reshape(coupled, (coupled.shape[0] * coupled.shape[1], ))
# print(coupled.shape)
# print(uncoupled.shape)
# sns.distplot(uncoupled, norm_hist=True)
# sns.distplot(coupled, norm_hist=True)
# plt.legend(['uncoupled', 'coupled'])
# plt.show()
# exit()

ratio = difference.row_ratio(timestamp, dbdt)
avg = difference.mvg_avg(pd.DataFrame(dbdt), 11)
var = difference.mvg_var(pd.DataFrame(dbdt), 11)
data_visualize.plotDat(timestamp, var, lbl)
plt.yscale('log')
plt.title('var vs Time')
plt.xlim((0.0014 + 4.30404e4, 0.0028 + 4.30404e4))
plt.show()
exit()
# super = slope
super = np.concatenate((avg, var, slope, rvalue, ratio), axis=1)
lbl = lbl[~np.isnan(super[:, 0])]
super = super[~np.isnan(super[:, 0])]
print(super.shape)

# lstm_mini(super, lbl)
# rnd_forest(timestamp, super, lbl, 0.20, 10000)
# PC_anal.PCA_custom(super, lbl, "super")
exit()

avg = difference.mvg_avg(pd.DataFrame(dbdt), 11)
data_visualize.plotDat(timestamp, avg, lbl)
plt.yscale('log')
data_visualize.plotDat(timestamp, avg, lbl)
plt.yscale('log')

# remove NAN values, stemming from windowing
lbl = lbl[~np.isnan(avg[:, 0])]
avg = avg[~np.isnan(avg[:, 0])]
# rnd_forest(timestamp, dbdt, lbl, 0.30)
# plt.show()
rnd_forest(timestamp, avg, lbl, 0.30)


def plotNormalnratio():
    fname = "../data/20171101_RAW_export.xyz"
    _, dbdt, lbl, timestamp = load_data2(fname, 8, 20)

    ratio = difference.row_ratio(timestamp, dbdt)
    data_visualize.plotDat(timestamp, ratio, lbl)
    data_visualize.plotDat(timestamp, dbdt, lbl)
    plt.yscale('log')
    plt.show()


def scatterPlt(x, y):
    # setup figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('slope', fontsize=15)
    ax.set_ylabel('r**2', fontsize=15)
    ax.set_title('Scatt: ', fontsize=20)
    # Set target ie. inuse or not, color and shape for data
    targets = [0, 1]
    colors = ['r', 'g']
    shapes = ['o', '^']
    # iterate over a tuple (zip), scatter plot 0 and 1
    for target, color, shape in zip(targets, colors, shapes):
        indicesToKeep = lbl == target
        ax.scatter(x[indicesToKeep]
                   , y[indicesToKeep]
                   , c=color
                   , marker=shape
                   , s=10)
    ax.legend(['coupled', 'non-coupled'])
    ax.grid()
    plt.xscale('log')
    plt.show()

# # TESTing the plot
# print(dbdt.shape)
# print(lbl.shape)
# dbdt = np.array([[1, 2], [1, 2], [1, 2],[1, 2],[1, 2],[1, 2]])
# print(dbdt.shape)
# lbl= np.array([1, 0, 1, 1, 1, 1])
# print(lbl.shape)
# timestamp = np.array([[1], [2], [3], [4], [5], [6]])

# df2 = pd.DataFrame({'a':[1, 2, 3, 4, 5, 6],
#                    'b':[4, 5, 6, 7, 8, 9],
#                    'c':[7, 8, 9, 10, 11, 12]})

# PC_anal.PCA_custom(dbdt, lbl, 'hej')
# exit()
# #normalize (min max)
# print("Min-max norm:")
# X_minmax = (dbdt - np.amin(dbdt)) / (np.amax(dbdt) - np.amin(dbdt))
# SVM_1.SVM_classify(X_minmax,lbl)
# print("")

# #standardize (z-score)
# dbdt_scaler = preprocessing.StandardScaler().fit(dbdt)
# dbdt_std = dbdt_scaler.transform(dbdt)
# X_std = dbdt_std
# print("z score norm:")
# SVM_1.SVM_classify(X_std,lbl)
# print("")

# df2 = pd.DataFrame({'a':[1, 2, 3, 4, 5, 6],
#                    'b':[4, 5, 6, 7, 8, 9],
#                    'c':[7, 8, 9, 10, 11, 12]})
# lr, rl = difference.mvg_var_leftright(df2, 3)