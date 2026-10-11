# IS 665 Group 3 - Student Version (v2): Memory-malware detection with logistic regression

A small, easy-to-read project. **One model only: logistic regression.**
The report is made by filling in the professor's own Word template, so its formatting is unchanged.

## How to run
1. Install Python packages:  `pip install -r requirements.txt`
2. Download the CIC-MalMem-2022 CSV from Kaggle and put it in `data/` (name: `Obfuscated-MalMem2022.csv`).
3. Run everything:  `python run_all.py`   (or run the steps one by one)

## The 5 steps (each file is short and has comments)
| File | What it does |
|---|---|
| `helpers.py` | Small shared functions (load data, read the Category text, make 7 ratio columns) |
| `step1_check_data.py` | Data-quality checks: missing values, duplicates, constant columns, outliers, labels, related dumps |
| `step2_explore_data.py` | Two charts (class balance, best columns) |
| `step3_prepare_and_split.py` | Cleaning, new columns, Split A (random) and Split B (hide whole malware families) |
| `step4_logistic_regression.py` | Trains logistic regression; recall, precision, false alarms, PR-AUC, 95% ranges, calibration, per-family recall |


## Words you need
- **Memory dump** = a saved copy of a computer's memory at one moment. One row = one dump.
- **Feature / column** = one number describing a dump.
- **Recall** = share of real malware we catch. **False alarm rate** = share of safe dumps we wrongly flag.
- **Validation part** = data used to check the model. **Test part** = locked until the final model.
- **Split B** = whole malware families are hidden from training, so the model is tested on malware it has never seen.

Outputs: `results/` (numbers as JSON, charts, data dictionary), `data/processed/` (cleaned data and splits).
The test part is never used in this deliverable.
