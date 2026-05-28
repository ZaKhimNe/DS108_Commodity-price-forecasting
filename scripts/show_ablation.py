import os, json

BASE_FILES = [
    "integrated_coffee_daily.csv",
    "integrated_coffee_weekly.csv",
    "integrated_corn_daily.csv",
    "integrated_corn_weekly.csv",
]

LGBM_DIR     = "models/14_lgbm_baseline"
ABLATION_DIR = "models/ablation"

def load_results(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        raw = json.load(f)
    return raw.get("splits", raw)

exp = {}
for fname in BASE_FILES:
    tag = fname.replace(".csv", "")
    exp.setdefault(fname, {})
    exp[fname]["A"] = load_results(os.path.join(LGBM_DIR, f"results_{tag}.json"))
    exp[fname]["B"] = load_results(os.path.join(ABLATION_DIR, "calendar_B", f"results_{tag}_llm_only.json"))
    exp[fname]["C"] = load_results(os.path.join(ABLATION_DIR, "calendar_C", f"results_{tag}_hybrid.json"))

print(f"{'Dataset':<30} {'Exp':<16} {'AUC-ROC':>8} {'PR-AUC':>8} {'F1':>7}")
print("-" * 73)
for fname in BASE_FILES:
    tag = fname.replace("integrated_", "").replace(".csv", "")
    for label, key in [("A_synthetic", "A"), ("B_llm_only", "B"), ("C_hybrid", "C")]:
        t   = exp[fname][key].get("test", {})
        auc = t.get("auc_roc") or 0
        pa  = t.get("pr_auc")  or 0
        f1  = t.get("f1")      or 0
        print(f"{tag:<30} {label:<16} {auc:>8.4f} {pa:>8.4f} {f1:>7.4f}")
    print()
