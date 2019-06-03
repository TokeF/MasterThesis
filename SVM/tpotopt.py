import sys
sys.path.append("..") # Adds higher directory to python modules path.
from tpot import TPOTClassifier
import tpot
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from utilities.data_reader import load_data2
from xgboost import XGBClassifier

fname = "../data/20171101_RAW_export.xyz"
df, dbdt, lbl, timestamp = load_data2(fname, 8, 23)
X_train, X_test, y_train, y_test = train_test_split(dbdt, lbl,
                                                    train_size=0.75, test_size=0.25)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.fit_transform(X_test)

# # apply random undersampling
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# rus = RandomUnderSampler(return_indices=True)
# X_train, y_train, id_rus = rus.fit_sample(X_train, y_train)

pipeline_optimizer = TPOTClassifier(generations=300, population_size=100, cv=5,
                                    random_state=42, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')
