from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from utilities.data_reader import load_data2

def select(X, y):
    clf = Pipeline([
      ('feature_selection', SelectFromModel(RandomForestClassifier())),
      ('classification', RandomForestClassifier())
    ])
    clf.fit(X, y)
    pred = clf.predict(X)
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(lbl, pred))
    print(classification_report(lbl, pred))

fname = "../data/20171101_RAW_export.xyz"
# fname = "../data/stendalmark_20181120_RAW_export.xyz"
df, dbdt, lbl, timestamp = load_data2(fname, 8, 23)
X = dbdt
y = lbl
select(X, y)