import sys

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, ShuffleSplit, cross_validate

from NN.window import build_array_skytem

sys.path.insert(0, '../utilities')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utilities.data_reader import timestampToTime
import utilities.data_visualize as dv
import utilities.difference

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def rnd_forest(timestamp: object, dbdt: object, lbl: object, tSize: object = 0.20, n_trees: object = 100,
               removelow: object = False) -> object:
    # X = list(chunks(range(len(lbl)), 5))
    # lbl2 = list(chunks(range(len(lbl)), 5))

    X = range(dbdt.shape[0])
    # X_train_idx, X_test_idx, lbl_train, lbl_test = train_test_split(X, lbl, test_size = tSize)
    #
    # X_train = dbdt[X_train_idx, :]
    # X_test = dbdt[X_test_idx, :]


    # #For segmenting the data
    # X_train_idx = [item for sublist in X_train_idx for item in sublist] #list comprehension :O
    # X_test_idx = [item for sublist in X_test_idx for item in sublist]
    # lbl_train_idx = [item for sublist in lbl_train_idx for item in sublist]
    # lbl_test_idx = [item for sublist in lbl_test_idx for item in sublist]
    # lbl_train = lbl[lbl_train_idx]
    # lbl_test = lbl[lbl_test_idx]
    #
    r = int(np.ceil(0.8 * dbdt.shape[0]))
    X_train = dbdt[0:r, :]
    lbl_train = lbl[0:r]
    X_train_idx = X[0:r]
    X_test = dbdt[r:, :]
    lbl_test = lbl[r:]
    X_test_idx = X[r:]
    #
    # X_train = np.transpose(build_array_skytem(15, X_train))
    # X_test = np.transpose(build_array_skytem(15, X_test))
    # lbl_train = lbl_train[7:-7]
    # lbl_test = lbl_test[7:-7]

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # apply random undersampling
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler()
    # X_train, lbl_train = rus.fit_sample(X_train, lbl_train)


    # Make classification
    classifier = RandomForestClassifier(n_estimators=n_trees)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=2, stop=1000, num=10)]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'min_samples_split': min_samples_split}
    # classifier = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
    #                                 random_state=42, n_jobs=-1)

    classifier.fit(X_train, lbl_train)
    lbl_scor = classifier.predict_proba(X_test)
    # lbl_pred = classifier.predict(X_test)
    if (removelow):
        # remove low precision
        lbl_scorA = np.asarray(lbl_scor)
        b = (lbl_scorA[:, 1] < 0.3) | (lbl_scorA[:, 1] > 0.7)
        pred = lbl_scorA[b, 1].round()
        test = lbl_test[b]
        print(len(pred) / len(lbl_test))
        #metrics
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(test,pred))
        print(classification_report(test,pred))
        exit()

    #find misclassified samples
    lbl_pred = lbl_scor[:,1] >= 0.1
    misclassified = np.where(lbl_test != lbl_pred)
    corclassified = np.where(lbl_test == lbl_pred)

    #plot data and red bars where data is misclassified
    timestamp = timestampToTime(timestamp)
    utilities.data_visualize.plotDat(timestamp, dbdt, lbl)
    plt.yscale('log')
    for xc in misclassified[0]:
        ogmark = timestamp[X_test_idx[xc]]
        plt.axvline(x=ogmark, color = 'red')
    # #plot correctly classified
    # for xc in corclassified[0]:
    #     ogmark = timestamp[X_test_idx[xc]]
    #     plt.axvline(x=ogmark, color = 'blue')


    #metrics
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve
    print(confusion_matrix(lbl_test,lbl_pred))
    report = classification_report(lbl_test, lbl_pred, output_dict=False)
    print(report)
    print("Test labels: " + str(lbl_test))
    print("predicted labels: " + str(lbl_pred))

    auc = dv.plot_roc(lbl_test, lbl_scor[:,1])
    print("AUC: " + str(auc))
    acc = sum(lbl_pred == lbl_test) / len(lbl_test)
    print("ACC: " + str(acc))
    np.save("rndScore", np.array([lbl_test, lbl_scor[:, 1]]))

    # print(report.get("0").get("recall"))
    # scoring = {'accuracy': make_scorer(accuracy_score),
    #            'precision': make_scorer(precision_score, pos_label = 0),
    #            'recall': make_scorer(recall_score, pos_label=0),
    #            'f1_score': make_scorer(f1_score)}
    # cv = ShuffleSplit(n_splits=5, test_size=0.2)
    # scores = cross_validate(classifier, dbdt, lbl, cv=cv, scoring=scoring, return_train_score=False)
    # print(scores)
    # print("macc: " + str(scores.get("test_accuracy").mean()))
    # print("mpre: " + str(scores.get("test_precision").mean()))
    # print("mrec: " + str(scores.get("test_recall").mean()))
    return lbl_scor, classifier.feature_importances_, report, acc





# ## Isolation forest
# uncoupled = dbdt.loc[df['DBDT_INUSE_Ch2GT14'] == 1].values
# coupled = dbdt.loc[df['DBDT_INUSE_Ch2GT14'] == 0].values
# print(coupled.shape[0] / uncoupled.shape[0])
# X_trainU, X_testU = train_test_split(uncoupled, test_size = 0.20)

# # Feature Scaling
# sc2 = StandardScaler()  
# X_trainU = sc2.fit_transform(X_trainU)  
# X_testU = sc2.transform(X_testU)

# sc3 = StandardScaler()  
# y_c = sc2.transform(coupled)

# from sklearn.ensemble import IsolationForest
# clf = IsolationForest(n_estimators = 1000, contamination = 0.13)
# clf.fit(X_train)
# # predictions
# X_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(y_c)
# print("Accuracy test:", list(X_pred_test).count(1)/X_pred_test.shape[0])
# print("Accuracy outlie: ", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
