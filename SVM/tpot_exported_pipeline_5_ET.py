import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

def pipe5ET(training_features, testing_features, training_target, testing_target):
    # Average CV score on the training set was:0.9804587264465511
    exported_pipeline = make_pipeline(
        make_union(
            FastICA(tol=0.35000000000000003),
            make_union(
                make_pipeline(
                    make_union(
                        make_pipeline(
                            Normalizer(norm="l1"),
                            RBFSampler(gamma=0.1)
                        ),
                        FunctionTransformer(copy)
                    ),
                    StandardScaler()
                ),
                StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=2, min_samples_leaf=2, min_samples_split=15))
            )
        ),
        ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.9000000000000001, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
    )

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)

    # metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(testing_target, results))
    print(classification_report(testing_target, results))