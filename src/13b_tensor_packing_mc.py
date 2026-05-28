import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import joblib
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "integrated")

# TARGET_MODE: "multiclass_soft"
# Output dir: data/tensors_multiclass_soft/
TARGET_MODE = "multiclass_soft"
TARGET_COL_MAP = {
    "binary":          "target_binary",
    "soft":            "target_soft",
    "regression":      "target_reg",
    "multiclass_soft": ["target_mc_down", "target_mc_flat", "target_mc_up"],
}
_TENSOR_DIR_MAP = {
    "binary":          "tensors",
    "soft":            "tensors_soft",
    "regression":      "tensors_reg",
    "multiclass_soft": "tensors_multiclass_soft",
}
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", _TENSOR_DIR_MAP[TARGET_MODE])

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# Cột hard label dùng cho feature selection (LightGBM cần scalar y)
_MC_HARD_COL = "target_multiclass"


def create_sliding_window_hybrid(dynamic_data, static_data, target, window_size):
    """
    X shape: (samples, window_size, features)
    y shape: (samples,) hoặc (samples, 3) khi multiclass_soft
    """
    X_dyn, X_stat, y = [], [], []

    for i in range(len(dynamic_data) - window_size + 1):
        X_dyn.append(dynamic_data[i : (i + window_size), :])

        if static_data is not None and static_data.shape[1] > 0:
            X_stat.append(static_data[i + window_size - 1, :])

        y.append(target[i + window_size - 1])

    X_stat_arr = np.array(X_stat) if len(X_stat) > 0 else np.array([])

    return np.array(X_dyn), X_stat_arr, np.array(y)


def lgbm_null_importance_selection(df, name, target_col, feat_sel_col, iv_threshold=0.0):
    """Sàng lọc Đặc trưng bằng LightGBM Null Importances. feat_sel_col luôn là scalar."""
    print(f"\n[+] Đang chạy LightGBM Null Importances cho {name}...")
    df_model = df.copy()

    all_protected = ['target', 'target_binary', 'target_soft', 'target_reg',
                     'target_mc_down', 'target_mc_flat', 'target_mc_up', 'target_multiclass',
                     'return_future', 'Close', 'currency_adjusted_close', 'Date']
    if isinstance(target_col, list):
        all_protected += target_col
    else:
        all_protected.append(target_col)

    features = [c for c in df_model.select_dtypes(include=[np.number]).columns
                if c not in all_protected]

    if len(df_model) == 0:
        print(f"   [CẢNH BÁO] Tập dữ liệu {name} bị RỖNG!")
        return df_model

    if len(features) == 0:
        print(f"   [CẢNH BÁO] Không còn đặc trưng nào trong {name}!")
        return df_model

    X = df_model[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_model[feat_sel_col]  # hard integer labels

    n_splits = min(3, len(X) - 1)
    if n_splits < 2:
        print(f"   [CẢNH BÁO] Dữ liệu quá ít ({len(X)} dòng).")
        return df_model

    tscv = TimeSeriesSplit(n_splits=n_splits)

    actual_imp = np.zeros(len(features))
    for train_idx, _ in tscv.split(X):
        model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        actual_imp += model.feature_importances_ / tscv.n_splits

    null_runs = 5
    null_imp = np.zeros(len(features))
    rng = np.random.default_rng(42)
    for _ in range(null_runs):
        y_shuffled = rng.permutation(y)
        fold_null = np.zeros(len(features))
        for train_idx, _ in tscv.split(X):
            model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
            model.fit(X.iloc[train_idx], y_shuffled[train_idx])
            fold_null += model.feature_importances_ / tscv.n_splits
        null_imp += fold_null / null_runs

    max_imp = actual_imp.max() + 1e-5
    iv_scores = np.maximum(0, actual_imp - null_imp) / max_imp

    imp_df = pd.DataFrame({'feature': features, 'IV_score': iv_scores})
    kept_features = imp_df[imp_df['IV_score'] >= iv_threshold]['feature'].tolist()

    if len(kept_features) == 0:
        print(f"    -> [CẢNH BÁO] IV=0, giữ toàn bộ {len(features)} biến.")
        kept_features = features

    keep_cols = set(all_protected) | set(kept_features)
    final_cols = [c for c in df_model.columns if c in keep_cols]

    print(f"    -> Đã giữ lại {len(kept_features)}/{len(features)} đặc trưng (IV >= {iv_threshold}).")
    return df_model[final_cols]


def process_module_5(file_name, window_sizes=[14, 21, 30], static_cols_input=[]):
    print(f"\n>>> Đang xử lý: {file_name}")
    file_path = os.path.join(INPUT_DIR, file_name)
    df = pd.read_csv(file_path)

    _all_target_cols = ['target', 'target_binary', 'target_soft', 'target_reg',
                        'target_mc_down', 'target_mc_flat', 'target_mc_up', 'target_multiclass']
    metadata_cols = ['Date', 'return_future', 'Close', 'currency_adjusted_close'] + _all_target_cols
    metadata_cols = [c for c in metadata_cols if c in df.columns]

    target_col = TARGET_COL_MAP["multiclass_soft"]  # list of 3 cols
    feat_sel_col = _MC_HARD_COL  # hard int label for LightGBM feature selection

    # Kiểm tra các cột multiclass tồn tại
    missing = [c for c in target_col + [feat_sel_col] if c not in df.columns]
    if missing:
        print(f"   [CẢNH BÁO] Thiếu cột multiclass: {missing}. Bỏ qua {file_name}.")
        return

    horizon    = 7 if 'daily' in file_name else 1
    test_start = int(len(df) * 0.8)
    val_start  = int(len(df) * 0.7)

    train_df = df.iloc[:val_start  - horizon].copy()
    val_df   = df.iloc[val_start  :test_start - horizon].copy()
    test_df  = df.iloc[test_start :].copy()

    train_df = lgbm_null_importance_selection(
        train_df, name=file_name, target_col=target_col,
        feat_sel_col=feat_sel_col, iv_threshold=0.05
    )
    val_df  = val_df[train_df.columns]
    test_df = test_df[train_df.columns]

    static_cols  = [c for c in static_cols_input if c in train_df.columns]
    dynamic_cols = [c for c in train_df.columns
                    if c not in metadata_cols + list(target_col) + [feat_sel_col] + static_cols]

    print(f"   - Đặc trưng Động (Dynamic): {len(dynamic_cols)} cột")
    print(f"   - Đặc trưng Tĩnh (Static): {len(static_cols)} cột")

    # Phân phối lớp (hard label)
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split_df[feat_sel_col].value_counts().sort_index()
        total  = len(split_df)
        dist   = {int(k): f"{v/total:.1%}" for k, v in counts.items()}
        print(f"   - {split_name}: {dist}  (n={total})")

    if len(dynamic_cols) > 0:
        scaler_dyn = MinMaxScaler()
        train_dyn_scaled = scaler_dyn.fit_transform(train_df[dynamic_cols])
        val_dyn_scaled   = scaler_dyn.transform(val_df[dynamic_cols])
        test_dyn_scaled  = scaler_dyn.transform(test_df[dynamic_cols])
        joblib.dump(scaler_dyn, os.path.join(OUTPUT_BASE_DIR,
                    f"scaler_dyn_{file_name.replace('.csv', '')}.joblib"))
    else:
        train_dyn_scaled = np.empty((len(train_df), 0))
        val_dyn_scaled   = np.empty((len(val_df),   0))
        test_dyn_scaled  = np.empty((len(test_df),  0))

    if len(static_cols) > 0:
        scaler_stat = MinMaxScaler()
        train_stat_scaled = scaler_stat.fit_transform(train_df[static_cols])
        val_stat_scaled   = scaler_stat.transform(val_df[static_cols])
        test_stat_scaled  = scaler_stat.transform(test_df[static_cols])
        joblib.dump(scaler_stat, os.path.join(OUTPUT_BASE_DIR,
                    f"scaler_stat_{file_name.replace('.csv', '')}.joblib"))
    else:
        train_stat_scaled = np.empty((len(train_df), 0))
        val_stat_scaled   = np.empty((len(val_df),   0))
        test_stat_scaled  = np.empty((len(test_df),  0))

    # Targets: shape (N, 3) — soft distributions
    train_target = train_df[list(target_col)].values  # (N, 3)
    val_target   = val_df[list(target_col)].values
    test_target  = test_df[list(target_col)].values

    for w in window_sizes:
        print(f"   [+] Đóng gói Window Size = {w}...")
        scenario_dir = os.path.join(OUTPUT_BASE_DIR, f"win_{w}", file_name.replace(".csv", ""))
        os.makedirs(scenario_dir, exist_ok=True)

        X_train_dyn, X_train_stat, y_train = create_sliding_window_hybrid(
            train_dyn_scaled, train_stat_scaled, train_target, w)
        X_val_dyn,   X_val_stat,   y_val   = create_sliding_window_hybrid(
            val_dyn_scaled,   val_stat_scaled,   val_target,   w)
        X_test_dyn,  X_test_stat,  y_test  = create_sliding_window_hybrid(
            test_dyn_scaled,  test_stat_scaled,  test_target,  w)

        assert len(X_train_dyn) == len(y_train), f"Lỗi lệch pha Train Dynamic tại W={w}"
        assert y_train.shape[1] == 3, f"y_train phải có 3 cột (N,3), got {y_train.shape}"

        np.save(os.path.join(scenario_dir, "X_train_dynamic.npy"), X_train_dyn)
        np.save(os.path.join(scenario_dir, "X_val_dynamic.npy"),   X_val_dyn)
        np.save(os.path.join(scenario_dir, "X_test_dynamic.npy"),  X_test_dyn)

        if len(static_cols) > 0:
            np.save(os.path.join(scenario_dir, "X_train_static.npy"), X_train_stat)
            np.save(os.path.join(scenario_dir, "X_val_static.npy"),   X_val_stat)
            np.save(os.path.join(scenario_dir, "X_test_static.npy"),  X_test_stat)

        np.save(os.path.join(scenario_dir, "y_train.npy"), y_train)  # (N, 3)
        np.save(os.path.join(scenario_dir, "y_val.npy"),   y_val)
        np.save(os.path.join(scenario_dir, "y_test.npy"),  y_test)

        train_df.iloc[w-1:].to_parquet(os.path.join(scenario_dir, "train_metadata.parquet"))
        val_df.iloc[w-1:].to_parquet(os.path.join(scenario_dir,   "val_metadata.parquet"))
        test_df.iloc[w-1:].to_parquet(os.path.join(scenario_dir,  "test_metadata.parquet"))

    print(f"   -> Saved to {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    files_to_process = [
        "integrated_coffee_daily.csv",
        "integrated_coffee_weekly.csv",
        "integrated_corn_daily.csv",
        "integrated_corn_weekly.csv"
    ]

    my_static_features = ['cal_sin_week', 'cal_cos_week', 'cal_sin_month', 'cal_cos_month',
                          'cal_is_harvest', 'cal_is_planting']

    for file in files_to_process:
        if "daily" in file:
            process_module_5(file, window_sizes=[14, 30, 45], static_cols_input=my_static_features)
        else:
            process_module_5(file, window_sizes=[4, 8, 12],  static_cols_input=my_static_features)

    print("\n[V] TENSOR PACKING MULTICLASS SOFT HOÀN TẤT!")
    print(f"    Output: data/tensors_multiclass_soft/  — y shape (N, 3) [down, flat, up]")
