import matplotlib
import matplotlib.pyplot as plt
plt.style.use('globalplotstyle') #custom layout of plots.
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score
import datetime
# have no idea what this does, but it is related to the timestamp conversion. Provokes a warning if removed
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def plotDat(timestamp : np.ndarray, dbdt : np.ndarray, lbl : np.ndarray):
    # dbdt = np.sign(dbdt) * np.log(np.abs(dbdt))

    #find index of label change from 0 to 1 or 1 to 0. Ved at finde indeks for en inbyrdes non-zero differens
    zero_crossings = np.where(np.diff(np.transpose(lbl).tolist()))[0]
    # if len(zero_crossings) == 0: print('no label change'), exit() #plot does not support no label change

    #genereate colors for plotting lines, so each gate is same color
    with plt.rc_context({'ytick.left':True}):
        fig, ax = plt.subplots()
        ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0.0,1.0,dbdt.shape[1])))
        ax.spines["left"].set_visible(True)
        # timestamp = timestampToTime(timestamp)

        #PLOT: the zerocross index, is the index before the change
        l = 0.3
        m = 6
        if len(zero_crossings) != 0:
            for i in range(len(zero_crossings) + 1): #always one for segment than zero changes
                #indice for first segment
                if i == 0:
                    startIdx = 0
                    endIdx = zero_crossings[i] + 1 + 1
                    coupled = lbl[zero_crossings[i]]
                #indice for middle segment
                if 0 < i < len(zero_crossings):
                    startIdx = zero_crossings[i - 1] + 1
                    endIdx = zero_crossings[i] + 1 + 1
                    coupled = lbl[zero_crossings[i]]
                #indice for last segment
                if i == len(zero_crossings):
                    startIdx = zero_crossings[i - 1] + 1
                    endIdx = None
                    coupled = not lbl[zero_crossings[i - 1]]
                #plot in colour or black if inuse or not
                if coupled:
                    plt.plot(timestamp[startIdx : endIdx], dbdt[startIdx : endIdx, :],'.-', linewidth=l, markersize=m)
                else:
                    plt.plot(timestamp[startIdx : endIdx], dbdt[startIdx : endIdx, :],'.-', color = 'black', linewidth=l, markersize=m)
        else:
            print('no label change')
            startIdx = 0
            endIdx = dbdt.shape[0] - 1
            coupled = lbl[0]
            if coupled:
                plt.plot(timestamp[startIdx: endIdx], dbdt[startIdx: endIdx, :], '.-', linewidth=l, markersize=m)
            else:
                plt.plot(timestamp[startIdx: endIdx], dbdt[startIdx: endIdx, :], '.-', color='black', linewidth=l,
                         markersize=m)

        if type(timestamp[0]) is datetime.datetime:
            time_formatter = matplotlib.dates.DateFormatter("%H:%M:%S")
            plt.gca().xaxis.set_major_formatter(time_formatter)

    plt.xlabel(r'Time [s]')
    plt.ylabel(r'$\mathrm{d}B/\mathrm{d}t$ $[V/Am^2]$')


def plotDat2(timestamp : np.ndarray, dbdt : np.ndarray, lbl : np.ndarray):
    #setup figure
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('jkas: ', fontsize = 20)

    #Set target ie. inuse or not, color and shape for data
    targets = [0, 1]
    colors = ['r', 'g']
    shapes = ['o', '^']
    print(dbdt.shape)
    print(timestamp.shape)
    #iterate over a tuple (zip), scatter plot 0 and 1
    for target, color, shape in zip(targets, colors, shapes):
       indicesToKeep = lbl == target
       print(indicesToKeep[:, 0].shape)
       ax.plot(timestamp[indicesToKeep[:, 0]]
                ,dbdt[indicesToKeep[:, 0], :]
                , c = color
                , marker = shape)
    ax.legend(['coupled', 'non-coupled'])
    ax.grid()

def histograms(dbdt, lbl, nbins = None, kdebool = True, normhist = True):
    #get indices for subplot
    features = dbdt.shape[1]
    rows = 2#max(1, np.floor(np.sqrt(features / 2)))
    cols = 2#np.ceil(np.sqrt(features))
    #find coupled and non coupled data
    indicesToKeep = lbl == 1
    fig = plt.figure()
    j = 1
    for i in range(features):
        #split into 2nd figure if we have >4 gates
        # if i == int(features / 2) and features/2 > 4:
        if i > 0 and i % 4 == 0:
            fig = plt.figure()
            j = 1
        fig.add_subplot(rows, cols, j)
        uncoupled = dbdt[indicesToKeep, i]
        coupled = dbdt[~indicesToKeep, i]
        sns.distplot(uncoupled, nbins, rug=True, kde=kdebool, norm_hist=normhist)
        sns.distplot(coupled, nbins, rug=True, kde=kdebool, norm_hist=normhist)
        plt.xlim(np.amin(dbdt[:,i]), np.amax(dbdt[:,i]))
        plt.title("gate " + str(i + 1))
        plt.legend(['uncoupled', 'coupled'])
        j += 1

def plot_training(history):
    plt.figure()
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])

    plt.figure()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])


def plot_misclassified(timestamp, X_test, y_test, y_pred):
    # plot the data
    plotDat(timestamp, X_test, y_test)
    plt.yscale('log')
    #find the misclassified soundings. Plot a vertical line at he location
    misclassified = np.where(y_test != y_pred)
    for xc in misclassified[0]:
        ogmark = timestamp[xc]
        plt.axvline(x=ogmark, color = 'red')


def plot_roc(y_true, y_score, ax, i, pos):
    if pos==0:
        y_score=-y_score
    fpr, tpr, th = roc_curve(y_true, y_score, pos_label=1)
    AUC = roc_auc_score(y_true, y_score)
    leg = "Iteration: " + str(i+1) + ", AUC: " + str('{0:.2f}'.format(AUC))
    ax.plot(fpr, tpr, label=leg)
    # ax.plot(fpr, tpr, fpr, th, label=leg)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])

    return AUC, tpr


def dist_score(score, ax, i=0):
    leg = "Iteration: " + str(i + 1)
    ax.hist(score, bins=int(np.sqrt(len(score))), weights=np.ones(len(score)) / len(score),
            label=leg, histtype='step', zorder=3)
    ax.grid(zorder=0)
    # ax.xaxis.grid(False, which='both')

