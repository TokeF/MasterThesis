from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def metrics(y_true, y_score, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    print(CM)
    report = classification_report(y_true, y_pred, output_dict=True)

    if sum(y_score > 1) != 0: #change score so smallest is best for auto
        y_score = -y_score
    AUC = roc_auc_score(y_true, y_score)
    print("AUC: " + str(AUC))
    ACC = sum(y_pred == y_true) / len(y_true)
    print("ACC: " + str(ACC))
    return report, CM, ACC, AUC