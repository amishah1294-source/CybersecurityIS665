"""STEP 3 - Clean the data, make new columns, and split into training / validation / test.
We make TWO splits:
  Split A = random split (what the proposal planned).
  Split B = hide whole malware families, so the model is tested on malware it has never seen."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from helpers import *

df = load_data()
info = add_label_info(df)
X = df[get_feature_columns(df)].copy()
p = {"rows_in": len(X)}

# --- 1. Fix bad values (this data has none, but we keep the checks)
p["missing_cells_filled"] = int(X.isna().sum().sum()); X = X.fillna(0)
p["negative_cells_fixed"] = int((X < 0).sum().sum()); X = X.clip(lower=0)

# --- 2. Remove duplicate rows and contradicting rows (BEFORE splitting, so copies cannot land in two parts)
is_dup = pd.concat([X, info["is_malware"]], axis=1).duplicated()
p["duplicates_removed"] = int(is_dup.sum())
X, info = X[~is_dup], info[~is_dup]
contradict = X.duplicated(keep=False)           # same numbers but a different label
p["contradicting_rows_removed"] = int(contradict.sum())
X, info = X[~contradict], info[~contradict]

# --- 3. Remove malware samples that appear under two different family names (confusing labels)
mal = info[info.is_malware == 1]
families_per_sample = mal.groupby("sample_id")["family"].nunique()
confusing = families_per_sample[families_per_sample > 1].index
drop_rows = info.index[(info.is_malware == 1) & info["sample_id"].isin(confusing)]
p["confusing_samples_removed"], p["confusing_rows_removed"] = len(confusing), len(drop_rows)
X, info = X.drop(index=drop_rows), info.drop(index=drop_rows)

# --- 4. Remove columns that never change
constant = [c for c in X.columns if X[c].nunique() <= 1]
p["constant_columns_removed"] = constant
X = X.drop(columns=constant)

# --- 5. Make 7 new ratio columns
X, ratio_formulas = add_ratio_features(X)
p["new_columns"] = ratio_formulas
p["rows_out"] = len(X)

# --- 6. Split A: random 70 / 15 / 15, keeping the same mix of types in each part
ids = info.index.values
train_ids, rest_ids = train_test_split(ids, test_size=0.30, random_state=SEED, stratify=info["type"])
val_ids, test_ids = train_test_split(rest_ids, test_size=0.50, random_state=SEED, stratify=info.loc[rest_ids, "type"])
split_a = pd.Series("train", index=info.index)
split_a[val_ids], split_a[test_ids] = "val", "test"

# --- 7. Split B: hide whole families. Safe rows are split randomly; for each malware type one family goes to
#        validation and one to test, the rest stay in training.
split_b = pd.Series("train", index=info.index)
safe_ids = info.index[info.is_malware == 0].values
_, rest_safe = train_test_split(safe_ids, test_size=0.30, random_state=SEED)
val_safe, test_safe = train_test_split(rest_safe, test_size=0.50, random_state=SEED)
split_b[val_safe], split_b[test_safe] = "val", "test"
rng = np.random.RandomState(SEED)
held_out = {"val": [], "test": []}
for malware_type, rows in info[info.is_malware == 1].groupby("type"):
    families = sorted(rows["family"].unique())
    rng.shuffle(families)
    split_b[rows.index[rows["family"] == families[0]]] = "test"
    split_b[rows.index[rows["family"] == families[1]]] = "val"
    held_out["test"].append(families[0]); held_out["val"].append(families[1])
p["held_out_families"] = held_out
p["split_a_sizes"], p["split_b_sizes"] = split_a.value_counts().to_dict(), split_b.value_counts().to_dict()

# --- 8. How much do training and validation share the same malware samples?
for name, split in (("a", split_a), ("b", split_b)):
    train_samples = set(info.loc[split[split == "train"].index].query("is_malware == 1")["sample_id"])
    val_malware = info.loc[split[split == "val"].index].query("is_malware == 1")
    p[f"split_{name}_shared_sample_pct"] = float(val_malware["sample_id"].isin(train_samples).mean() * 100)

# --- 9. Remove columns that are almost copies of another column (correlation > 0.98).
#        We look only at rows that are in TRAINING in both splits, so no validation/test data is used.
both_train = X.loc[(split_a == "train") & (split_b == "train")]
corr = both_train.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
near_copies = [c for c in upper.columns if (upper[c] > 0.98).any()]
X = X.drop(columns=near_copies)
p["near_copy_columns_removed"] = near_copies
p["columns_left"] = X.shape[1]

X.to_pickle("data/processed/features.pkl")
info.to_pickle("data/processed/labels.pkl")
pd.DataFrame({"split_a": split_a, "split_b": split_b}).to_csv("data/processed/splits.csv")
save_json(p, "prep.json")
print({k: v for k, v in p.items() if not isinstance(v, (list, dict))})
print("Split A:", p["split_a_sizes"], "| Split B:", p["split_b_sizes"], "| hidden families:", held_out)
print("Shared samples (validation vs training): Split A", round(p["split_a_shared_sample_pct"], 1), "%  Split B", round(p["split_b_shared_sample_pct"], 1), "%")
