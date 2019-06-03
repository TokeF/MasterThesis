import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy


def pipe4bernoulliET(training_features, testing_features, training_target, testing_target):
    # Average CV score on the training set was:0.9731889330896907
    exported_pipeline = make_pipeline(
        make_union(
            FunctionTransformer(copy),
            make_pipeline(
                PCA(iterated_power=3, svd_solver="randomized"),
                MinMaxScaler()
            )
        ),
        StackingEstimator(estimator=BernoulliNB(alpha=1.0, fit_prior=False)),
        ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.7500000000000001, min_samples_leaf=2, min_samples_split=3, n_estimators=100)
    )

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)

    # metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(testing_target, results))
    print(classification_report(testing_target, results))
