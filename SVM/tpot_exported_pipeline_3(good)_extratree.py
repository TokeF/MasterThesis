import sys
sys.path.append("..") # Adds higher directory to python modules path.
import numpy as np
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, Normalizer, PolynomialFeatures
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1).values
# training_features, testing_features, training_target, testing_target = \
#             train_test_split(features, tpot_data['target'].values, random_state=42)
from utilities.data_reader import load_data2
fname = "../data/vildbjerg_20171101_modified_RAW_export.xyz"
# fname = "../data/stendalmark_20181120_RAW_export.xyz"
df, dbdt, lbl, timestamp = load_data2(fname, 8, 23)
training_features, testing_features, training_target, testing_target = \
            train_test_split(dbdt, lbl, random_state=42)

# apply random undersampling
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, CondensedNearestNeighbour
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN
rus = RandomOverSampler()
# training_features, training_target = rus.fit_sample(training_features, training_target)

from SVM.tpot_exported_pipeline_4_bernoulliEXTT import pipe4bernoulliET
from SVM.tpot_exported_pipeline_5_ET import pipe5ET
pipe4bernoulliET(training_features, testing_features, training_target, testing_target)
exit()

# Average CV score on the training set was:0.9438983551659608
exported_pipeline = make_pipeline(
    make_union(
        make_union(
            Normalizer(norm="l1"),
            PCA(iterated_power=4, svd_solver="randomized")
        ),
        FunctionTransformer(copy)
    ),
    MinMaxScaler(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.7500000000000001, min_samples_leaf=2, min_samples_split=4, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

#metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(testing_target, results))
print(classification_report(testing_target, results))

# scores = cross_val_score(exported_pipeline, dbdt, lbl, cv=5)
# print(scores)