import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
def classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred=(y_prob>=threshold).astype(int)
    p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0)
    try: auc=roc_auc_score(y_true,y_prob)
    except Exception: auc=float('nan')
    return dict(precision=float(p), recall=float(r), f1=float(f1), auc=float(auc))
