from pathlib import Path
import h5py
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path("final_publication_results")
TABLES = ROOT / "tables"
H5 = ROOT / "features.h5"
SPLITS = TABLES / "splits.csv"
THRESHOLD = 0.75

def compute_metrics(y_true, y_prob, model_name):
    y_pred = (y_prob >= THRESHOLD).astype(int)

    return {
        "Model": model_name,
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "TP": int(((y_pred == 1) & (y_true == 1)).sum()),
        "TN": int(((y_pred == 0) & (y_true == 0)).sum()),
        "FP": int(((y_pred == 1) & (y_true == 0)).sum()),
        "FN": int(((y_pred == 0) & (y_true == 1)).sum()),
        "threshold": THRESHOLD,
    }

splits = pd.read_csv(SPLITS)

train_idx = splits.loc[splits["split"] == "train", "index"].to_numpy()
test_idx = splits.loc[splits["split"] == "test", "index"].to_numpy()

with h5py.File(H5, "r") as f:
    X = f["qc_features"][:]
    y = f["labels"][:].astype(int)

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

print("Train:", X_train.shape, "Positives:", int((y_train == 1).sum()), "Negatives:", int((y_train == 0).sum()))
print("Test:", X_test.shape, "Positives:", int((y_test == 1).sum()), "Negatives:", int((y_test == 0).sum()))

models = {
    "Logistic_regression_QC": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")
    ),
    "Random_forest_QC": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "Gradient_boosting_QC": GradientBoostingClassifier(
        random_state=42
    ),
}

results = []

for name, model in models.items():
    print("\nTraining:", name)
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
        y_prob = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)

    row = compute_metrics(y_test, y_prob, name)
    results.append(row)
    print(row)

out = TABLES / "baseline_comparison_results.csv"
df = pd.DataFrame(results)
df.to_csv(out, index=False)

print("\nBaseline results:")
print(df.to_string(index=False))
print("Saved:", out)
