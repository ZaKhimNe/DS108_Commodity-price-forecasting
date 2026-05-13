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
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "tensors")

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

def create_sliding_window_hybrid(dynamic_data, static_data, target, window_size):
    """
    Chuyển đổi mảng 2D thành 3D Tensor (Sliding Window)
    X shape: (samples, window_size, features)
    y shape: (samples,)
    """
    X_dyn, X_stat, y = [], [], []
    
    for i in range(len(dynamic_data) - window_size + 1):
        X_dyn.append(dynamic_data[i : (i + window_size), :])
        
        if static_data is not None and static_data.shape[1] > 0:
            X_stat.append(static_data[i + window_size - 1, :])
            
        y.append(target[i + window_size - 1])
    
    X_stat_arr = np.array(X_stat) if len(X_stat) > 0 else np.array([])
        
    return np.array(X_dyn), X_stat_arr, np.array(y)


def lgbm_null_importance_selection(df, name, target_col='target', iv_threshold=0.0):
    """Sàng lọc Đặc trưng bằng LightGBM Null Importances & TimeSeriesSplit"""
    print(f"\n[+] Đang chạy LightGBM Null Importances cho {name}...")
    df_model = df.copy()
    
    protected_cols = [target_col, 'return_future', 'Close', 'currency_adjusted_close']
    features = [c for c in df_model.select_dtypes(include=[np.number]).columns if c not in protected_cols]
    
    if len(df_model) == 0:
        print(f"   [CẢNH BÁO] Tập dữ liệu {name} bị RỖNG (0 dòng) trước khi vào LightGBM!")
        return df_model 
        
    if len(features) == 0:
        print(f"   [CẢNH BÁO] Không còn đặc trưng (features) nào trong {name} để học!")
        return df_model
    
    X = df_model[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_model[target_col]
    
    n_splits = min(3, len(X) - 1) 
    if n_splits < 2:
        print(f"   [CẢNH BÁO] Dữ liệu {name} quá ít ({len(X)} dòng), không đủ để chia TimeSeriesSplit.")
        return df_model
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    actual_imp = np.zeros(len(features))
    for train_idx, val_idx in tscv.split(X):
        model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        actual_imp += model.feature_importances_ / tscv.n_splits
        
    null_runs = 5 
    null_imp = np.zeros(len(features))
    for _ in range(null_runs):
        y_shuffled = np.random.permutation(y)
        fold_null = np.zeros(len(features))
        for train_idx, val_idx in tscv.split(X):
            model = lgb.LGBMClassifier(n_estimators=50, random_state=42, verbose=-1)
            model.fit(X.iloc[train_idx], y_shuffled[train_idx])
            fold_null += model.feature_importances_ / tscv.n_splits
        null_imp += fold_null / null_runs
        
    max_imp = actual_imp.max() + 1e-5
    iv_scores = np.maximum(0, actual_imp - null_imp) / max_imp
    
    imp_df = pd.DataFrame({'feature': features, 'IV_score': iv_scores})
    kept_features = imp_df[imp_df['IV_score'] >= iv_threshold]['feature'].tolist()
    
    if len(kept_features) == 0:
        print(f"    -> [CẢNH BÁO] Không có biến nào đạt IV >= {iv_threshold}. Tự động giữ lại toàn bộ {len(features)} biến gốc!")
        kept_features = features

    final_cols = [c for c in df_model.columns if c in protected_cols + kept_features]
    
    print(f"    -> Đã giữ lại {len(kept_features)}/{len(features)} đặc trưng (IV >= {iv_threshold}).")
    return df_model[final_cols]

def process_module_5(file_name, window_sizes=[14, 21, 30], static_cols_input=[]):
    print(f"\n>>> Đang xử lý: {file_name}")
    file_path = os.path.join(INPUT_DIR, file_name)
    df = pd.read_csv(file_path)
    
    df = lgbm_null_importance_selection(df, name=file_name, target_col='target', iv_threshold=0.0)
    
    metadata_cols = ['Date', 'return_future', 'Close', 'currency_adjusted_close']
    metadata_cols = [c for c in metadata_cols if c in df.columns]
    
    target_col = 'target'
    static_cols = [c for c in static_cols_input if c in df.columns]
    dynamic_cols = [c for c in df.columns if c not in metadata_cols + [target_col] + static_cols]

    print(f"   - Đặc trưng Động (Dynamic): {len(dynamic_cols)} cột")
    print(f"   - Đặc trưng Tĩnh (Static): {len(static_cols)} cột")
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    if len(dynamic_cols) > 0:
        scaler_dyn = MinMaxScaler()
        train_dyn_scaled = scaler_dyn.fit_transform(train_df[dynamic_cols])
        test_dyn_scaled = scaler_dyn.transform(test_df[dynamic_cols])
        joblib.dump(scaler_dyn, os.path.join(OUTPUT_BASE_DIR, f"scaler_dyn_{file_name.replace('.csv', '')}.joblib"))
    else:
        train_dyn_scaled = np.empty((len(train_df), 0))
        test_dyn_scaled = np.empty((len(test_df), 0))
    
    if len(static_cols) > 0:
        scaler_stat = MinMaxScaler()
        train_stat_scaled = scaler_stat.fit_transform(train_df[static_cols])
        test_stat_scaled = scaler_stat.transform(test_df[static_cols])
        joblib.dump(scaler_stat, os.path.join(OUTPUT_BASE_DIR, f"scaler_stat_{file_name.replace('.csv', '')}.joblib"))
    else:
        train_stat_scaled = np.empty((len(train_df), 0))
        test_stat_scaled = np.empty((len(test_df), 0))
    
    train_target = train_df[target_col].values
    test_target = test_df[target_col].values
    
    for w in window_sizes:
        print(f"   [+] Đóng gói Window Size = {w}...")
        scenario_dir = os.path.join(OUTPUT_BASE_DIR, f"win_{w}", file_name.replace(".csv", ""))
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Đóng gói Hybrid
        X_train_dyn, X_train_stat, y_train = create_sliding_window_hybrid(train_dyn_scaled, train_stat_scaled, train_target, w)
        X_test_dyn, X_test_stat, y_test = create_sliding_window_hybrid(test_dyn_scaled, test_stat_scaled, test_target, w)
        
        # Kiểm tra lệch pha
        assert len(X_train_dyn) == len(y_train), f"Lỗi lệch pha Train Dynamic tại W={w}"
        if len(static_cols) > 0:
            assert len(X_train_stat) == len(y_train), f"Lỗi lệch pha Train Static tại W={w}"
            
        # 5. Xuất file Numpy
        np.save(os.path.join(scenario_dir, "X_train_dynamic.npy"), X_train_dyn)
        np.save(os.path.join(scenario_dir, "X_test_dynamic.npy"), X_test_dyn)
        
        if len(static_cols) > 0:
            np.save(os.path.join(scenario_dir, "X_train_static.npy"), X_train_stat)
            np.save(os.path.join(scenario_dir, "X_test_static.npy"), X_test_stat)
            
        np.save(os.path.join(scenario_dir, "y_train.npy"), y_train)
        np.save(os.path.join(scenario_dir, "y_test.npy"), y_test)
        
        train_df.iloc[w-1:].to_parquet(os.path.join(scenario_dir, "train_metadata.parquet"))
        test_df.iloc[w-1:].to_parquet(os.path.join(scenario_dir, "test_metadata.parquet"))

if __name__ == "__main__":
    files_to_process = [
        "integrated_coffee_daily.csv",
        "integrated_coffee_weekly.csv",
        "integrated_corn_daily.csv",
        "integrated_corn_weekly.csv"
    ]
    
    my_static_features = ['week_of_year', 'week_cos', 'month_sin', 'month_cos', 'cal_harvesting', 'cal_planting']
    
    for file in files_to_process:
        if "daily" in file:
            process_module_5(file, window_sizes=[14, 30, 45], static_cols_input=my_static_features)
        else:
            process_module_5(file, window_sizes=[4, 8, 12], static_cols_input=my_static_features)
            
    print("\n[V] MODULE 5 HOÀN TẤT: Đã bóc tách thành công luồng Dynamic (3D) và Static (2D)!")