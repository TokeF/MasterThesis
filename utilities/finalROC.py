import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
plt.style.use('globalplotstyle') #custom layout of plots.

rf = np.load("rf_score.npy")
cnn = np.load("cnn_score.npy")
auto = np.load("auto_score.npy")

name = ["Random forest", "1D CNN", "Autoencoder"]

fig, ax = plt.subplots()
ax.set_prop_cycle('color', plt.cm.tab10(range(0,5)))
for n, i in enumerate([rf, cnn, auto]):
    # because i appended the score and label wrong, and i did not ahve time to change it
    split = int(i.shape[0] / 2)
    pred = i[0:split]
    lbl = i[split:]
    if n == 2:
        pred = -1*pred
    fpr, tpr, th = roc_curve(lbl, pred, pos_label=1)
    AUC = roc_auc_score(lbl, pred)
    leg = "AUC: " + str('{0:.2f}'.format(AUC)) + ", "  + name[n]
    ax.plot(fpr, tpr, label=leg)

ax.legend()
ax.plot([0,1], [0,1], color="tab:gray", linestyle = "-.", label="50% line")
ax.set_ylabel('True positive rate'); ax.set_xlabel('False positive rate')
plt.show()