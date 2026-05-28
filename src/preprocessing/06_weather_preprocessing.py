import pandas as pd
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

raw_weather_path = os.path.join(PROJECT_ROOT, "data", "preprocessed", "weather", "weather_despiked")
processed_path = os.path.join(PROJECT_ROOT, "data", "preprocessed", "weather", "weather_clean")
os.makedirs(processed_path, exist_ok=True)

def preprocess_weather_file(file_name):    
    df = pd.read_csv(os.path.join(raw_weather_path, file_name))
    
    # Đồng bộ timezone utc=True để chuẩn hóa dữ liệu ngày tháng
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date').set_index('date')

    print(f"\n--- Missing values trong {file_name} ---")
    print(df.isnull().sum())

    weather_cols = ['temperature_2m_max', 'temperature_2m_min', 'vpd_max', 'et0_fao_evapotranspiration']
    existing_cols = [col for col in weather_cols if col in df.columns]
    
    if existing_cols:
        df[existing_cols] = df[existing_cols].ffill()

    # Xử lý lượng mưa (điền 0 nếu không có mưa)
    if 'precipitation_sum' in df.columns:
        df['precipitation_sum'] = df['precipitation_sum'].fillna(0)
        
    # Tích hợp TOÀN BỘ các đặc trưng trượt (Rolling Features) phong phú
    if 'temperature_2m_max' in df.columns:
        df['temp_max_rolling_7d'] = df['temperature_2m_max'].rolling(window=7).mean()
        df['temp_max_rolling_14d'] = df['temperature_2m_max'].rolling(window=14).mean()
        df['temp_max_cumsum_30d'] = df['temperature_2m_max'].rolling(window=30).sum()
        df['temp_max_diff_1d'] = df['temperature_2m_max'].diff()
        df['temp_max_std_7d'] = df['temperature_2m_max'].rolling(window=7).std()
    
    if 'precipitation_sum' in df.columns:
        df['precip_sum_rolling_7d'] = df['precipitation_sum'].rolling(window=7).sum()
        df['precip_sum_rolling_14d'] = df['precipitation_sum'].rolling(window=14).sum()
        df['precip_cumsum_30d'] = df['precipitation_sum'].rolling(window=30).sum()
        df['precip_diff_1d'] = df['precipitation_sum'].diff()
        df['precip_std_7d'] = df['precipitation_sum'].rolling(window=7).std()
        df['dry_days_7d'] = (df['precipitation_sum'].rolling(window=7).sum() == 0).astype(int)
    
    if 'et0_fao_evapotranspiration' in df.columns:
        df['et0_rolling_7d'] = df['et0_fao_evapotranspiration'].rolling(window=7).mean()
        df['et0_cumsum_30d'] = df['et0_fao_evapotranspiration'].rolling(window=30).sum()

    # Loại bỏ các hàng NaN sinh ra do cửa sổ trượt và reset index
    df = df.dropna()
    df = df.reset_index()
    return df

if __name__ == "__main__":
    print("BẮT ĐẦU QUÉT THƯ MỤC...")

    if not os.path.exists(raw_weather_path):
        print(f"Không tìm thấy thư mục đầu vào: {raw_weather_path}")
    else:
        for file_name in os.listdir(raw_weather_path):
            if file_name.endswith(".csv"):
                cleaned_df = preprocess_weather_file(file_name)
                
                save_name = f"clean_{file_name}"
                save_path = os.path.join(processed_path, save_name)
                
                cleaned_df.to_csv(save_path, index=False)
                print(f">> Đã xử lý và lưu thành công: {save_name}")

        print("\nHOÀN THÀNH TOÀN BỘ!")