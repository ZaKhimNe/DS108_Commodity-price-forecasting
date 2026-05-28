import pandas as pd
import os

# Sử dụng Đường dẫn tuyệt đối động (Script-relative path)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Giả định script nằm ở src/preprocessing/, lùi 2 cấp sẽ ra thư mục gốc
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

daily_weather_path = os.path.join(PROJECT_ROOT, "data", "preprocessed", "weather", "weather_clean")
weekly_weather_path = os.path.join(PROJECT_ROOT, "data", "preprocessed", "weather", "weather_weekly")

os.makedirs(weekly_weather_path, exist_ok=True)

def resample_to_weekly(file_name):
    df = pd.read_csv(os.path.join(daily_weather_path, file_name))
    
    # Đồng bộ với timezone utc=True từ các file tiền xử lý trước đó
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.set_index('date')

    agg_dict = {}
    
    # Tích hợp toàn bộ các cột từ file copy
    mean_cols = [
        'temperature_2m_max', 'temperature_2m_min', 
        'et0_fao_evapotranspiration', 'vpd_max',
        'temp_max_rolling_7d', 'temp_max_rolling_14d', 'temp_max_std_7d',
        'precip_sum_rolling_7d', 'precip_sum_rolling_14d', 'precip_std_7d',
        'et0_rolling_7d'
    ]
    for col in mean_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'
            
    sum_cols = ['precipitation_sum']
    for col in sum_cols:
        if col in df.columns:
            agg_dict[col] = 'sum'

    last_cols = ['dry_days_7d', 'temp_max_cumsum_30d', 'precip_cumsum_30d', 'et0_cumsum_30d']
    for col in last_cols:
        if col in df.columns:
            agg_dict[col] = 'last'

    # Gộp theo tuần
    df_weekly = df.resample('W-MON').agg(agg_dict)
    df_weekly = df_weekly.dropna()
    df_weekly = df_weekly.reset_index()
    
    return df_weekly

if __name__ == "__main__":
    print("BẮT ĐẦU CHUYỂN ĐỔI SANG DỮ LIỆU TUẦN...")

    if not os.path.exists(daily_weather_path):
        print(f"Không tìm thấy thư mục đầu vào: {daily_weather_path}")
    else:
        for file_name in os.listdir(daily_weather_path):
            if file_name.endswith(".csv") and file_name.startswith("clean_"):
                df_weekly = resample_to_weekly(file_name)
                
                save_name = file_name.replace("clean_", "weekly_")
                save_path = os.path.join(weekly_weather_path, save_name)
                
                df_weekly.to_csv(save_path, index=False)
                print(f">> Đã chuyển sang data tuần và lưu: {save_name}")

        print("\nHOÀN THÀNH ĐỒNG BỘ MỨC ĐỘ CHI TIẾT (WEEKLY)!")