"""Small helper functions used by all the steps (kept simple on purpose)."""
import json
import numpy as np
import pandas as pd

DATA_FILE = "data/Obfuscated-MalMem2022.csv"   # put the Kaggle CSV here
SEED = 42                                       # fixed random seed -> same results every run


def load_data():
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip() for c in df.columns]
    return df


def get_feature_columns(df):
    """Every column except the two label columns is a feature (a number about the memory dump)."""
    return [c for c in df.columns if c not in ("Category", "Class")]


def add_label_info(df):
    """The 'Category' text looks like  Ransomware-Conti-<hash>-3.raw
    We split it into: type, family, sample id (the hash) and dump number."""
    cat = df["Category"].astype(str)
    info = pd.DataFrame(index=df.index)
    info["is_malware"] = (df["Class"] != "Benign").astype(int)
    parts = cat.str.split("-")
    info["type"] = np.where(info["is_malware"] == 1, parts.str[0].str.capitalize(), "Benign")
    info["family"] = np.where(info["is_malware"] == 1, info["type"] + "-" + parts.str[1].fillna("Unknown"), "Benign")
    found = cat.str.extract(r"-([0-9a-fA-F]{32,64})-(\d+)\.raw$")
    info["sample_id"] = np.where(info["is_malware"] == 1, found[0].fillna(cat), "Benign")
    return info


def safe_divide(a, b):
    result = a / b.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def add_ratio_features(X):
    """Make 7 new 'ratio' columns from existing ones. Returns the new table and a list of (name, formula)."""
    X = X.copy()
    formulas = []

    def add(name, values, formula):
        X[name] = values
        formulas.append((name, formula))

    add("inj_per_proc", safe_divide(X["malfind.ninjections"], X["pslist.nproc"]), "malfind.ninjections / pslist.nproc")
    add("hidden_dll_ratio", safe_divide(X["ldrmodules.not_in_load"], X["dlllist.ndlls"]), "ldrmodules.not_in_load / dlllist.ndlls")
    for name in ("nfile", "nmutant", "nkey"):
        add(f"handles_{name[1:]}_frac", safe_divide(X[f"handles.{name}"], X["handles.nhandles"]), f"handles.{name} / handles.nhandles")
    not_in_cols = [c for c in X.columns if c.startswith("psxview.not_in_") and "false" not in c]
    add("psxview_hidden_per_proc", safe_divide(X[not_in_cols].sum(axis=1), X["pslist.nproc"]), "sum(psxview.not_in_*) / pslist.nproc")
    add("svc_driver_frac", safe_divide(X["svcscan.kernel_drivers"] + X["svcscan.fs_drivers"], X["svcscan.nservices"]),
        "(svcscan.kernel_drivers + svcscan.fs_drivers) / svcscan.nservices")
    return X, formulas


def save_json(data, name):
    with open("results/" + name, "w") as f:
        json.dump(data, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def load_json(name):
    with open("results/" + name) as f:
        return json.load(f)


# Plain-English meaning of each Volatility group and of each column (used for the data dictionary)
GROUP_MEANING = {
    "pslist": "Running programs (processes)",
    "dlllist": "DLL files (shared code libraries) loaded by programs",
    "handles": "Things programs have open: files, registry keys, locks, events, ports",
    "ldrmodules": "Loaded code missing from the normal lists (sign of hidden code)",
    "malfind": "Signs of injected code (malware pushing code into another program)",
    "psxview": "Programs seen in one process list but not another (sign of hiding)",
    "modules": "Kernel modules (core parts of the operating system)",
    "svcscan": "Windows background services and drivers",
    "callbacks": "Kernel callbacks (hooks into the operating system)",
}

FIELD_MEANING = {
    "pslist.nproc": "Number of running programs", "pslist.nppid": "Number of different parent programs",
    "pslist.avg_threads": "Average threads per program", "pslist.nprocs64bit": "Number of 64-bit programs",
    "pslist.avg_handlers": "Average handles per program",
    "dlllist.ndlls": "Total DLL files loaded", "dlllist.avg_dlls_per_proc": "Average DLL files per program",
    "handles.nhandles": "Total open handles", "handles.avg_handles_per_proc": "Average handles per program",
    "handles.nport": "Open port handles", "handles.nfile": "Open file handles", "handles.nevent": "Open event handles",
    "handles.ndesktop": "Open desktop handles", "handles.nkey": "Open registry-key handles",
    "handles.nthread": "Open thread handles", "handles.ndirectory": "Open directory handles",
    "handles.nsemaphore": "Open semaphore handles", "handles.ntimer": "Open timer handles",
    "handles.nsection": "Open memory-section handles", "handles.nmutant": "Open mutex (lock) handles",
    "ldrmodules.not_in_load": "Modules missing from the 'loaded' list", "ldrmodules.not_in_init": "Modules missing from the 'initialised' list",
    "ldrmodules.not_in_mem": "Modules missing from the 'in memory' list",
    "ldrmodules.not_in_load_avg": "Same as not_in_load, averaged per program", "ldrmodules.not_in_init_avg": "Same as not_in_init, averaged per program",
    "ldrmodules.not_in_mem_avg": "Same as not_in_mem, averaged per program",
    "malfind.ninjections": "Memory regions that look like injected code", "malfind.commitCharge": "Memory committed by those regions",
    "malfind.protection": "Regions that are readable, writable and executable", "malfind.uniqueInjections": "Different injections found",
    "modules.nmodules": "Number of kernel modules",
    "svcscan.nservices": "Number of Windows services", "svcscan.kernel_drivers": "Kernel driver services",
    "svcscan.fs_drivers": "File-system driver services", "svcscan.process_services": "Services that run as their own program",
    "svcscan.shared_process_services": "Services that share a program", "svcscan.interactive_process_services": "Services that can interact with the desktop",
    "svcscan.nactive": "Services that are active",
    "callbacks.ncallbacks": "Total kernel callbacks", "callbacks.nanonymous": "Callbacks with no owner name", "callbacks.ngeneric": "Generic callbacks",
}


def describe_field(name):
    if name in FIELD_MEANING:
        return FIELD_MEANING[name]
    if name.startswith("psxview.not_in_") and name.endswith("_false_avg"):
        return "Average version of the matching 'not seen' check"
    if name.startswith("psxview.not_in_"):
        return "Programs missing from the '" + name.replace("psxview.not_in_", "") + "' list"
    return "New ratio column made by us (see Section 5)"
