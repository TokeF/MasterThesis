import random
import sys
from sklearn.preprocessing import StandardScaler

from NN.window import build_array_skytem

sys.path.insert(0, '../utilities')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utilities.data_reader import load_data2, remove_edge
from utilities.calc_metrics import metrics
import utilities.data_visualize as dv
import utilities.difference

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def rnd_forest2(timestamp_val, dbdt, lbl, X_valOG, X_train, X_val,
                y_train, y_val, n_trees = 100):

    print("Training size: ", X_train.shape[0])
    print("Val size: ", X_val.shape[0])
    #Create Model
    classifier = RandomForestClassifier(n_estimators=n_trees, max_features=None)

    #Train
    classifier.fit(X_train, y_train)

    #Predict
    y_scor = classifier.predict_proba(X_val)

    #find misclassified samples
    y_pred = y_scor[:, 1] >= 0.8
    misclassified = np.where(y_val != y_pred)
    corclassified = np.where(y_val == y_pred)

    #plot data and red bars where data is misclassified
    # plt.figure("Soundings")
    # timestamp = timestampToTime(timestamp)

    dv.plot_misclassified(timestamp_val, X_valOG, y_val, y_pred)
    # utilities.data_visualize.plotDat(timestamp, dbdt, lbl)
    # plt.yscale('log')
    # timestamp = timestamp - timestamp[0]
    # for xc in misclassified[0]:
    #     ogmark = timestamp[test_idx[xc]]
    #     plt.axvline(x=ogmark, color = 'red')

    # #plot correctly classified
    # for xc in corclassified[0]:
    #     ogmark = timestamp[X_test_idx[xc]]
    #     plt.axvline(x=ogmark, color = 'blue')


    #metrics
    report, CM, ACC, AUC = metrics(y_val, y_scor[:,1], y_pred)
    return y_scor[:,1], classifier.feature_importances_, report, ACC, AUC, CM

def main():
    ## Data preprocessing
    fname0 = "../data/stendalmark_20181120_RAW_export.xyz"
    fname1 = "../data/stendalmark_20181121_RAW_export.xyz"

    df, dbdt, lbl, timestamp, gtimes = load_data2(fname0, 8, 24)
    dbdt, lbl, timestamp = remove_edge(timestamp, dbdt, lbl, 20)
    timestamp = (timestamp - timestamp[0]) * 10 ** 5

    df0, dbdt0, lbl0, timestamp0, gtimes0 = load_data2(fname1, 8, 24)
    dbdt0, lbl0, timestamp0 = remove_edge(timestamp0, dbdt0, lbl0, 20)
    timestamp0 = (timestamp0 - timestamp0[0]) * 10 ** 5 + timestamp[-1] + 0.7

    dbdt = np.concatenate((dbdt, dbdt0));
    lbl = np.concatenate((lbl, lbl0))
    timestamp = np.concatenate((timestamp, timestamp0))
    sc = StandardScaler()

    r = int(np.ceil(0.8 * dbdt.shape[0]))
    dbdt = dbdt[0:r, :]
    lbl = lbl[0:r]
    timestamp=timestamp[0:r]
    idx = range(0, dbdt.shape[0])
    v = int(np.ceil(0.8 * dbdt.shape[0]))
    X_test = dbdt[v:, :]
    X_train = dbdt[0:v, :]
    y_train = lbl[0:v]
    y_test = lbl[v:]
    idx_train = list(idx[0:v])
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # X_train = build_array_skytem(41, X_train)
    # X_test = build_array_skytem(41, X_test)
    # y_score, _, report, acc = rnd_forest2(timestamp, dbdt, lbl, idx[r:],
    #                                       X_train.T, X_test.T,
    #                                       y_train[20:-20], y_test[20:-20], n_trees=100)

    # trees = range(1,100)
    trees = np.linspace(0.01,1.0, 50)
    metric = np.zeros((len(trees),))
    for i, n_trees in enumerate(trees):
        # random.shuffle(idx_train)
        n = int(np.ceil(n_trees * X_train.shape[0]))
        _, _, report, acc, auc, CM = rnd_forest2(timestamp[v:], dbdt, lbl, X_test,
                                              X_train[idx_train[0:n],:], X_test,
                                              y_train[idx_train[0:n]], y_test, n_trees=100)
        metric[i] = CM[1,0] + CM[0,1]
        plt.figure()
        plt.close('all')
        print("Iteration: ", i)
    plt.plot(trees, metric)
    plt.ylabel("Total errors")
    plt.xlabel("Training set in use [Fraction]")
    plt.show()


if __name__ == '__main__':
    main()