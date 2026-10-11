"""STEP 2 - Explore the data with 2 charts (EDA = exploratory data analysis)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from helpers import *

COLORS = {"Benign": "#2b7bba", "Ransomware": "#d1495b", "Spyware": "#edae49", "Trojan": "#66a182"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

df = load_data()
info = add_label_info(df)
X = df[get_feature_columns(df)]
X = X.loc[:, X.nunique() > 1]          # ignore columns that never change
e = {}

# ---------- Figure 1: how many safe / malware rows, and how many of each type
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
counts = info["is_malware"].map({0: "Safe", 1: "Malware"}).value_counts()[["Safe", "Malware"]]
bars = ax[0].bar(counts.index, counts.values, color=["#2b7bba", "#d1495b"])
ax[0].bar_label(bars, labels=[f"{v:,}\n({v / len(df):.0%})" for v in counts.values], fontsize=8)
ax[0].set_title("Safe vs malware", fontweight="bold"); ax[0].set_ylabel("Memory dumps"); ax[0].set_ylim(0, counts.max() * 1.25)
type_counts = info["type"].value_counts()[["Benign", "Ransomware", "Spyware", "Trojan"]]
bars = ax[1].bar(type_counts.index, type_counts.values, color=[COLORS[t] for t in type_counts.index])
ax[1].bar_label(bars, labels=[f"{v:,}" for v in type_counts.values], fontsize=8)
ax[1].set_title("Dumps by type", fontweight="bold"); ax[1].set_ylim(0, type_counts.max() * 1.2); ax[1].tick_params(axis="x", labelsize=7)
fig.tight_layout(); fig.savefig("results/figures/fig1_class_balance.png"); plt.close(fig)

# ---------- Figure 2: the 4 columns that separate safe from malware best (histograms)
auc = {c: roc_auc_score(info.is_malware, X[c]) for c in X.columns}
auc = {c: max(a, 1 - a) for c, a in auc.items()}
top4 = sorted(auc, key=auc.get, reverse=True)[:4]
fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))
for ax_i, col in zip(axes, top4):
    values = np.log1p(X[col].clip(lower=0))
    low, high = values.quantile(0.005), values.quantile(0.995)
    bins = np.linspace(low, high if high > low else low + 1, 40)
    ax_i.hist(values[info.is_malware == 0].clip(low, high), bins=bins, density=True, alpha=0.6, color="#2b7bba", label="Safe")
    ax_i.hist(values[info.is_malware == 1].clip(low, high), bins=bins, density=True, alpha=0.6, color="#d1495b", label="Malware")
    ax_i.set_title(col.replace(".", ".\n", 1) + f"\nAUC={auc[col]:.3f}", fontsize=7.5, fontweight="bold")
    ax_i.set_yticks([]); ax_i.tick_params(labelsize=7)
axes[0].set_ylabel("Density"); axes[0].legend(fontsize=6.5, frameon=False)
fig.supxlabel("log(1 + column value)", fontsize=8)
fig.tight_layout(); fig.savefig("results/figures/fig2_best_columns.png"); plt.close(fig)

e["best_columns"] = {}
for col in top4:
    e["best_columns"][col] = {
        "auc": auc[col],
        "median_safe": float(X.loc[info.is_malware == 0, col].median()),
        "median_malware": float(X.loc[info.is_malware == 1, col].median()),
        "most_common_value_share_safe": float(X.loc[info.is_malware == 0, col].value_counts(normalize=True).iloc[0]),
        "most_common_value_share_malware": float(X.loc[info.is_malware == 1, col].value_counts(normalize=True).iloc[0]),
    }

save_json(e, "eda.json")
print({c: round(v["auc"], 3) for c, v in e["best_columns"].items()})
