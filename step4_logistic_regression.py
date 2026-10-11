"""STEP 4 - The baseline model: LOGISTIC REGRESSION (the only model in this version).
Logistic regression gives each column a weight, adds them up, and turns the total into a
probability that a memory dump is malware. We say 'malware' when the probability is 0.5 or more.
Everything is checked on the VALIDATION part. The test part stays locked."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from helpers import *

X = pd.read_pickle("data/processed/features.pkl")
info = pd.read_pickle("data/processed/labels.pkl")
splits = pd.read_csv("data/processed/splits.csv", index_col=0).loc[X.index]


def train_model(X_train, y_train):
    """log(1+x) squashes huge numbers, StandardScaler puts columns on one scale (learned from TRAINING rows only)."""
    logged = np.log1p(X_train.clip(lower=0))
    scaler = StandardScaler().fit(logged)
    model = LogisticRegression(max_iter=3000, random_state=SEED).fit(scaler.transform(logged), y_train)
    return scaler, model


def predict_probability(scaler, model, X_new, class_index=1):
    return model.predict_proba(scaler.transform(np.log1p(X_new.clip(lower=0))))[:, class_index]


def score_model(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "false_alarm_rate": fp / (fp + tn), "roc_auc": roc_auc_score(y, prob),
            "pr_auc": average_precision_score(y, prob), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "brier": float(np.mean((prob - y) ** 2)), "ece": calibration_error(y, prob)}


def calibration_error(y, prob, n_bins=10):
    """ECE: split predictions into 10 groups by confidence; compare confidence with the real share of malware."""
    y, prob = np.asarray(y), np.asarray(prob)
    group = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for g in range(n_bins):
        mask = group == g
        if mask.any():
            error += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(error)


def uncertainty_range(y, prob, sample_ids, n_draws=500):
    """95% range by re-drawing the data 500 times. Dumps of the same malware sample stay together
    (they are near-copies), so the range is not too narrow. Safe rows are drawn one by one."""
    rng = np.random.RandomState(SEED)
    y, prob = np.asarray(y), np.asarray(prob)
    groups = np.array([s if label == 1 else f"safe{i}" for i, (s, label) in enumerate(zip(sample_ids, y))])
    codes, _ = pd.factorize(groups)
    rows_of_group = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    results = {"recall": [], "false_alarm_rate": [], "f1": [], "pr_auc": []}
    for _ in range(n_draws):
        chosen = rng.randint(0, len(rows_of_group), len(rows_of_group))
        rows = np.concatenate([rows_of_group[g] for g in chosen])
        yy, pp = y[rows], prob[rows]
        pred = pp >= 0.5
        tp, fn = (pred & (yy == 1)).sum(), (~pred & (yy == 1)).sum()
        fp, tn = (pred & (yy == 0)).sum(), (~pred & (yy == 0)).sum()
        results["recall"].append(tp / (tp + fn)); results["false_alarm_rate"].append(fp / (fp + tn))
        results["f1"].append(2 * tp / (2 * tp + fp + fn)); results["pr_auc"].append(average_precision_score(yy, pp))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in results.items()}


out = {}
for name, column in (("split_a", "split_a"), ("split_b", "split_b")):
    train, val = splits[column] == "train", splits[column] == "val"
    scaler, model = train_model(X[train], info.is_malware[train])
    prob = predict_probability(scaler, model, X[val])
    y_val = info.is_malware[val]
    res = score_model(y_val, prob)
    res["range_95"] = uncertainty_range(y_val.values, prob, info.loc[val, "sample_id"].values)
    res["n_train"], res["n_val"] = int(train.sum()), int(val.sum())
    # which malware did we miss?
    missed = info[val][(y_val == 1) & (prob < 0.5)]
    res["missed_by_type"] = missed["type"].value_counts().to_dict()
    res["missed_by_family"] = missed["family"].value_counts().head(4).to_dict()
    # which columns matter most?
    weights = pd.Series(model.coef_[0], index=X.columns)
    res["top_columns"] = weights.reindex(weights.abs().sort_values(ascending=False).index).head(3).round(3).to_dict()
    # second job: name the malware TYPE (ransomware / spyware / trojan), using malware rows only
    m_train, m_val = train & (info.is_malware == 1), val & (info.is_malware == 1)
    type_scaler, type_model = train_model(X[m_train], info.type[m_train])
    type_pred = type_model.predict(type_scaler.transform(np.log1p(X[m_val].clip(lower=0))))
    biggest_type = info.type[m_train].value_counts().idxmax()
    res["type_accuracy"] = float(accuracy_score(info.type[m_val], type_pred))
    res["type_guessing_accuracy"] = float((info.type[m_val] == biggest_type).mean())
    out[name] = res

# Leave-one-family-out: hide each family in turn (uses only training + validation rows of Split B; the test rows stay locked)
use = splits["split_b"].isin(["train", "val"])
Xu, iu = X[use], info[use]
families = sorted(iu.loc[iu.is_malware == 1, "family"].unique())
rng = np.random.RandomState(SEED)
fold = pd.Series(-1, index=iu.index)
safe_rows = iu.index[iu.is_malware == 0]
fold[safe_rows] = rng.randint(0, len(families), len(safe_rows))
for i, fam in enumerate(families):
    fold[iu.family == fam] = i
prob_all = pd.Series(0.0, index=iu.index)
for i in range(len(families)):
    test_rows = fold == i
    hidden_samples = set(iu.loc[test_rows & (iu.is_malware == 1), "sample_id"])
    train_rows = (~test_rows) & ~iu["sample_id"].isin(hidden_samples)
    scaler, model = train_model(Xu[train_rows], iu.is_malware[train_rows])
    prob_all[test_rows] = predict_probability(scaler, model, Xu[test_rows])
lofo = score_model(iu.is_malware, prob_all)
lofo["range_95"] = uncertainty_range(iu.is_malware.values, prob_all.values, iu["sample_id"].values)
lofo["recall_by_family"] = {fam: {"recall": float((prob_all[iu.family == fam] >= 0.5).mean()), "n": int((iu.family == fam).sum())} for fam in families}
out["leave_one_family_out"] = lofo

out["goals"] = {"recall_goal": 0.80, "false_alarm_goal": 0.10, "type_accuracy_goal": 0.80,
                "recall_met": out["split_a"]["recall"] >= 0.80, "false_alarm_met": out["split_a"]["false_alarm_rate"] < 0.10,
                "type_accuracy_met": out["split_a"]["type_accuracy"] >= 0.80}
save_json(out, "metrics.json")

for name in ("split_a", "split_b", "leave_one_family_out"):
    r = out[name]
    print(f"{name:22s} recall {r['recall']:.4f}  false alarms {r['false_alarm_rate']:.4f}  F1 {r['f1']:.4f}  PR-AUC {r['pr_auc']:.4f}  brier {r['brier']:.4f}  ece {r['ece']:.4f}")
print("type accuracy:", round(out["split_a"]["type_accuracy"], 3), round(out["split_b"]["type_accuracy"], 3), "| guessing:", round(out["split_a"]["type_guessing_accuracy"], 3), round(out["split_b"]["type_guessing_accuracy"], 3))
print({f: round(v["recall"], 3) for f, v in lofo["recall_by_family"].items()})
