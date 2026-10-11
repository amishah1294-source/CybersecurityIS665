"""STEP 1 - Check the quality of the data.
Looks for: missing values, duplicates, impossible values, columns that never change, extreme values,
class balance, confusing labels, and 'related rows' (dumps of the same malware sample)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from helpers import *

df = load_data()
info = add_label_info(df)
feature_cols = get_feature_columns(df)
X = df[feature_cols]
r = {}   # r collects all results; it is saved at the end

r["filename"] = DATA_FILE.split("/")[-1]
r["n_rows"], r["n_raw_columns"], r["n_features"] = len(df), df.shape[1], len(feature_cols)

# 1. Missing and impossible values
r["missing_cells"] = int(X.isna().sum().sum())
r["infinite_cells"] = int(np.isinf(X).sum().sum())
r["negative_cells"] = int((X < 0).sum().sum())

# 2. Duplicate rows (same numbers AND same label) and contradictions (same numbers, different label)
same_row = pd.concat([X, info["is_malware"]], axis=1).duplicated()
r["duplicate_rows"] = int(same_row.sum())
r["duplicate_pct"] = float(same_row.mean() * 100)
r["duplicates_by_type"] = info.loc[same_row, "type"].value_counts().to_dict()
no_dups = pd.concat([X, info["is_malware"]], axis=1)[~same_row]
r["contradicting_rows"] = int(no_dups[feature_cols].duplicated(keep=False).sum())

# 3. Columns that never (or almost never) change
n_unique = X.nunique()
r["constant_columns"] = n_unique[n_unique <= 1].index.tolist()
r["almost_constant_columns"] = [c for c in feature_cols if n_unique[c] > 1 and X[c].value_counts(normalize=True).iloc[0] > 0.995]

# 4. Extreme values (outliers): share of rows outside 1.5 x IQR, for each column
q1, q3 = X.quantile(0.25), X.quantile(0.75)
iqr = q3 - q1
ok_cols = iqr > 0
outlier_share = (((X.loc[:, ok_cols] < (q1 - 1.5 * iqr)[ok_cols]) | (X.loc[:, ok_cols] > (q3 + 1.5 * iqr)[ok_cols])).mean() * 100)
r["outlier_median_pct"] = float(outlier_share.median())
r["columns_over_5pct_outliers"] = int((outlier_share > 5).sum())
r["very_lopsided_columns"] = int((X.skew().abs() > 2).sum())

# 5. Class balance
r["n_safe"] = int((info.is_malware == 0).sum())
r["n_malware"] = int((info.is_malware == 1).sum())
r["malware_pct"] = float(info.is_malware.mean() * 100)
r["type_counts"] = info["type"].value_counts().to_dict()

# 6. Labels: do Category and Class agree? Is the same malware sample listed under two families?
r["label_disagreements"] = int(((df["Category"].str.lower().str.startswith("benign")) != (df["Class"] == "Benign")).sum())
mal = info[info.is_malware == 1]
fam_per_sample = mal.groupby("sample_id")["family"].nunique()
r["samples_with_two_families"] = int((fam_per_sample > 1).sum())
r["rows_with_two_families"] = int(mal["sample_id"].isin(fam_per_sample[fam_per_sample > 1].index).sum())
r["families_per_type"] = mal.groupby("type")["family"].nunique().to_dict()
r["n_families"] = int(mal["family"].nunique())
r["biggest_family_pct"] = float(mal["family"].value_counts().iloc[0] / len(mal) * 100)
r["n_category_names"] = int(df["Category"].nunique())

# 7. What is one row? Count dumps per malware sample, and how similar dumps of one sample are
dumps_per_sample = mal["sample_id"].value_counts()
r["malware_samples"] = int(len(dumps_per_sample))
r["dumps_per_sample_min"], r["dumps_per_sample_median"], r["dumps_per_sample_max"] = (
    int(dumps_per_sample.min()), float(dumps_per_sample.median()), int(dumps_per_sample.max()))
logged = np.log1p(X.loc[mal.index].clip(lower=0))
logged = logged.loc[:, logged.std() > 0]
within_sample = logged - logged.groupby(mal["sample_id"].values).transform("mean")
r["within_sample_variation_pct"] = float((within_sample.var() / logged.var()).median() * 100)

# 6b. Columns that are almost copies of each other
corr = X.loc[:, n_unique > 1].corr().abs().values
r["near_copy_pairs"] = int((corr[np.triu_indices_from(corr, k=1)] > 0.98).sum())

# 8. Is any single column almost enough to tell safe from malware? (AUC: 1.0 = perfect, 0.5 = guessing)
auc = {c: roc_auc_score(info.is_malware, X[c]) for c in feature_cols if n_unique[c] > 1}
auc = {c: max(a, 1 - a) for c, a in auc.items()}
best = sorted(auc.items(), key=lambda kv: -kv[1])
r["best_columns"] = dict(best[:6])
r["best_auc"] = best[0][1]
r["columns_auc_over_95"] = int(sum(a > 0.95 for a in auc.values()))
r["has_time_column"] = any(w in c.lower().split(".") for c in df.columns for w in ("time", "date", "timestamp"))

save_json(r, "data_quality.json")

# Data dictionary: one line per column (used in the report appendix)
rows = []
for c in feature_cols:
    rows.append({"variable": c, "type": "Numeric (" + ("whole number" if (X[c] % 1 == 0).all() else "decimal") + ")",
                 "meaning": describe_field(c), "group": c.split(".")[0],
                 "min": X[c].min(), "median": X[c].median(), "max": X[c].max(),
                 "role": "Not used (never changes)" if c in r["constant_columns"] else "Feature (input)"})
pd.DataFrame(rows).to_csv("results/data_dictionary.csv", index=False)

for k, v in r.items():
    if not isinstance(v, (dict, list)):
        print(f"{k:32s} {v}")
print("types:", r["type_counts"], "| families per type:", r["families_per_type"])
print("Saved results/data_quality.json and results/data_dictionary.csv")
