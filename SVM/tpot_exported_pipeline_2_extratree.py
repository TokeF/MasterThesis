import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from utilities.data_reader import load_data2

# NOTE: Make sure that the class is labeled 'target' in the data file
fname = "../data/20171101_RAW_export.xyz"
# fname = "../data/stendalmark_20181120_RAW_export.xyz"
df, dbdt, lbl, timestamp = load_data2(fname, 8, 23)
training_features, testing_features, training_target, testing_target = \
            train_test_split(dbdt, lbl, test_size=0.20, random_state=42)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
rus = RandomUnderSampler(return_indices=True)
training_features, training_target, id_rus = rus.fit_sample(training_features, training_target)

# Average CV score on the training set was:0.9141662730278697
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    PCA(iterated_power=1, svd_solver="randomized"),
    ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.9500000000000001, min_samples_leaf=5, min_samples_split=8, n_estimators=10000)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(testing_target, results))
print(classification_report(testing_target, results))