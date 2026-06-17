import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

raw_weather_path = os.path.join(PROJECT_ROOT, "data", "raw", "weather")
despiked_path = os.path.join(PROJECT_ROOT, "data", "preprocessed", "weather", "weather_despiked")
os.makedirs(despiked_path, exist_ok=True)

def apply_miqr(df, column_name, window=15, k=3.0, sustained_days=7):
    """
    Áp dụng MIQR nhưng bảo toàn các tín hiệu khí hậu cực đoan kéo dài (hạn hán, sóng nhiệt).
    Chỉ xóa (đánh NaN) các gai nhiễu đơn lẻ (isolated spikes < sustained_days).
    """
    if column_name not in df.columns:
        return df

    Q1 = df[column_name].rolling(window=window, center=False, min_periods=1).quantile(0.25)
    Q3 = df[column_name].rolling(window=window, center=False, min_periods=1).quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    # Phát hiện tất cả các điểm vượt ngưỡng
    is_anomaly = (df[column_name] < lower_bound) | (df[column_name] > upper_bound)
    
    # Tách các đợt biến động liên tiếp thành các block
    blocks = (~is_anomaly).cumsum()
    # Đếm số ngày liên tục vượt ngưỡng trong mỗi block
    anomaly_counts = is_anomaly.groupby(blocks).transform('sum')
    
    # Phân loại: Gai nhiễu (dưới 7 ngày) và Sự kiện khí hậu thực sự (>= 7 ngày)
    isolated_spikes = is_anomaly & (anomaly_counts < sustained_days)
    sustained_events = is_anomaly & (anomaly_counts >= sustained_days)
    
    # Chỉ đánh NaN cho gai nhiễu đơn lẻ
    df.loc[isolated_spikes, column_name] = np.nan
    
    num_spikes = isolated_spikes.sum()
    num_sustained = sustained_events.sum()
    
    if num_spikes > 0 or num_sustained > 0:
        print(f"   -> Cột {column_name}: Chuyển {num_spikes} điểm nhiễu (Spike) thành NaN. "
              f"Bảo toàn {num_sustained} điểm thuộc sự kiện thời tiết kéo dài (>= {sustained_days} ngày).")
        
    return df

def apply_rolling_flatline_detection(df, column_name, window=5, std_threshold=1e-4, dead_days_threshold=3):
    if column_name not in df.columns or df[column_name].isnull().all():
        return df

    rolling_std = df[column_name].rolling(window=window, min_periods=1).std().fillna(0)
    is_dead = (rolling_std < std_threshold).astype(int)
    is_prolonged_dead = is_dead.rolling(window=dead_days_threshold, min_periods=1).sum() >= dead_days_threshold

    num_dead_points = is_prolonged_dead.sum()
    if num_dead_points > 0:
        print(f"   -> Cột {column_name}: Tìm thấy {num_dead_points} điểm kẹt cảm biến (Flatline) bằng Rolling Std.")
        df.loc[is_prolonged_dead, column_name] = np.nan

    return df

if __name__ == "__main__":
    print("BẮT ĐẦU QUÉT VÀ LOẠI BỎ NHIỄU (MIQR + WAVELET)...")

    if not os.path.exists(raw_weather_path):
        print(f"Không tìm thấy thư mục dữ liệu đầu vào: {raw_weather_path}")
    else:
        for file_name in os.listdir(raw_weather_path):
            if file_name.endswith(".csv"):
                print(f"\nĐang xử lý: {file_name}")
                file_path = os.path.join(raw_weather_path, file_name)
                df = pd.read_csv(file_path)
                
                # Tích hợp logic xử lý chuẩn hóa ngày tháng
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None).dt.normalize()
                
                continuous_cols = ['temperature_2m_max', 'temperature_2m_min', 'et0_fao_evapotranspiration', 'vpd_max']
                
                for col in continuous_cols:
                    # 1. Cắt nhiễu đột biến (sustained_days=7: bảo toàn đợt hạn hán >= 7 ngày)
                    df = apply_miqr(df, col, window=15, k=3.0, sustained_days=7)
                    
                    # 2. Xử lý lỗi kẹt cảm biến
                    df = apply_rolling_flatline_detection(df, col, window=5, std_threshold=1e-4, dead_days_threshold=4)
                
                save_path = os.path.join(despiked_path, f"despiked_{file_name}")
                df.to_csv(save_path, index=False)

        print("\nHOÀN THÀNH BƯỚC LOẠI BỎ NHIỄU!")