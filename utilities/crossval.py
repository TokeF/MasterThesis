import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler

from NN.AutoEnc2 import autoencoder2, autoencoder3
from NN.window import build_array
from utilities import data_visualize
from NN.NNdriver import runCNN

sys.path.append("..")  # Adds higher directory to python modules path.
from utilities.data_reader import load_data2, remove_edge
from SVM.rndForest2 import rnd_forest2

# from SVM.SVM_1 import SVM_classify
def autolabel(rects, ax, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

fname = "../data/vildbjerg_20171101_RAW_export.xyz"
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

lblOG = lbl
dbdtOG = dbdt

r = int(np.ceil(0.8 * dbdt.shape[0]))
timestamp_train = timestamp[0:r]
timestamp_test =  timestamp[r:]
X_trainFull = dbdt[0:r, :]
X_test = dbdt[r:, :]
X_testOG = X_test
y_trainFull = lbl[0:r]
y_test = lbl[r:]
# sc = StandardScaler()
# X_trainFull = sc.fit_transform(X_trainFull)
# X_test = sc.transform(X_test)

# timestamp = (timestamp - timestamp[0])*10**6
print("Coupled samples: ", dbdt[lbl == 0,:].shape[0] / dbdt[lbl == 1,:].shape[0])
print("coupled ",sum(y_test == 0) / X_test.shape[0])
print("total coupled ",X_test[y_test==0,:].shape[0])
print("total", X_test.shape[0])

predictor = "cnn"
splits = 5
kf = KFold(n_splits=splits, shuffle=False)
# if predictor == "rf":
#     kf = ShuffleSplit(n_splits=splits, test_size=0.2)

metrics = np.zeros([9,splits])
tpr = []
i = 0
width = 0.35; b=1
figROC, axROC = plt.subplots()
axROC.set_prop_cycle('color', plt.cm.tab10(range(0,5)))
figBAR, axBAR = plt.subplots()
axBAR.set_prop_cycle('color', plt.cm.Set2([0,1]))
axBAR.grid(zorder=0)
figDist, axDist = plt.subplots()
pos = 1 #for the ROC curve, autoencoder needs pos 0
print("model chosen: ", predictor)
test = True
for train_index, val_index in kf.split(X_trainFull):
    if test:
        print("Skip cross validation"); break
    print("TRAIN:", train_index, "VAL:", val_index)
    X_train, X_val = X_trainFull[train_index], X_trainFull[val_index]
    y_train, y_val = y_trainFull[train_index], y_trainFull[val_index]
    timestamp_val = timestamp_train[val_index]
    X_valOG = X_val

    valBar = axBAR.bar(b - width/2, 1.0 - sum(y_val==0) / sum(y_val==1), width, zorder=3)
    trainBar = axBAR.bar(b + width/2, 1.0 - sum(y_train==0) / sum(y_train==1), width, zorder=3)
    autolabel(valBar, axBAR)
    autolabel(trainBar, axBAR)
    print("Val: ", sum(y_val==0) / sum(y_val==1))
    print("Train: ", sum(y_train==0) / sum(y_train==1))
    b+=1

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    if predictor == "rf":
        y_score, _, report, acc, _,_ = rnd_forest2(timestamp_val, dbdt, lbl, X_valOG,
                                                 X_train, X_val, y_train, y_val)

    if (predictor == "cnn"):
        y_score, report, acc, _, _ = runCNN(timestamp, X_train, X_val, y_train, y_val,
                                      X_valOG, timestamp_val, build=True)
    if predictor == "auto":
        X_train = X_train[y_train == 1, :]
        y_score, report, acc, _, _ = autoencoder2(X_train, X_val, y_val, X_valOG, timestamp_val)
        pos = 0

    AUC, line = data_visualize.plot_roc(y_val, y_score, axROC, i, pos=pos)
    tpr.append(line)

    data_visualize.dist_score(y_score[y_score <= 1], axDist, i)
    print("AUC: ", AUC)

    metrics[0,i] = report.get("0").get("precision")
    metrics[1, i] = report.get("0").get("recall")
    metrics[2, i] = report.get("1").get("precision")
    metrics[3, i] = report.get("1").get("recall")
    metrics[4, i] = report.get("0").get("support")
    metrics[5, i] = report.get("1").get("support")
    metrics[6, i] = report.get("weighted avg").get("f1-score")
    metrics[7, i] = acc
    metrics[8, i] = AUC
    i+=1

######## print metrics of val
for m, std in zip(metrics.mean(axis=1), metrics.std(axis=1)):
    print(m, " +- ", std)

###################### Run test set
if test:
    print("Running test")
    if predictor == "auto":
        X_trainFull = X_trainFull[y_trainFull == 1, :]

    sc2 = StandardScaler()
    X_trainFull = sc2.fit_transform(X_trainFull)
    X_test = sc2.transform(X_test)
    if predictor == "rf":
        y_score2, _, report2, acc2, auc2, cm2 = rnd_forest2(timestamp_test, dbdt, lbl, X_testOG,
                                                 X_trainFull, X_test, y_trainFull, y_test)
    if (predictor == "cnn"):
        y_score2, report2, acc2,auc2,cm2 = runCNN(timestamp, X_trainFull, X_test, y_trainFull, y_test,
                                      X_testOG, timestamp_test, build=True)
    if predictor == "auto":
        y_score2, report2, acc2, auc2, cm2 = autoencoder2(X_trainFull, X_test, y_test, X_testOG, timestamp_test)
        pos = 0
    print("Final report: ", report2)
    print("Final ACC: ", acc2)
    print("Final AUC: ", auc2)
    print("bdfklbfskl ",roc_auc_score(y_test, y_score2))
    np.save(predictor + "_score", np.concatenate((y_score2, y_test)))

########################## Graphs of CV
# axDist.set_xlim([0, 1])
axDist.set_ylabel('Fraction of total counts')
if predictor=="auto":
    axDist.set_xlabel('Mean absolute error')
    axDist.legend(loc='upper right')
else:
    axDist.set_xlabel('Prediction')
    axDist.legend(loc='upper left')
th = 0.8
axDist.axvline(x=th, ymax=0.75, color='blue')
axDist.text(th*0.85, axDist.get_ylim()[1]*0.8, s="Threshold " + str(th))

# axROC.fill_between(tpr[0], tpr[1])
axROC.plot([0,1], [0,1], color="tab:gray", linestyle = "-.", label="50% line")
axROC.set_ylabel('True positive rate'); axROC.set_xlabel('False positive rate')
axROC.legend()
axBAR.legend(["Validation set", "Train set"])
axBAR.set_xlabel('Iteration no.'); axBAR.set_ylabel('Fraction of non coupled data')
axBAR.set_yticklabels([]); axBAR.set_facecolor('white')
print("model chosen: ", predictor)
plt.show()
