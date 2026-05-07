import pandas as pd
import os

daily_weather_path = "data/preprocessed/weather/weather_clean"
weekly_weather_path = "data/preprocessed/weather/weather_weekly"
os.makedirs(weekly_weather_path, exist_ok=True)

def resample_to_weekly(file_name):
    df = pd.read_csv(os.path.join(daily_weather_path, file_name))
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    agg_dict = {}
    
    mean_cols = [
        'temperature_2m_max', 'temperature_2m_min', 
        'et0_fao_evapotranspiration', 'vpd_max',
        'temp_max_rolling_7d', 'temp_max_rolling_14d', 'precip_sum_rolling_7d'
    ]
    for col in mean_cols:
        if col in df.columns:
            agg_dict[col] = 'mean'
            
    if 'precipitation_sum' in df.columns:
        agg_dict['precipitation_sum'] = 'sum'

    df_weekly = df.resample('W').agg(agg_dict)

    df_weekly = df_weekly.dropna()

    df_weekly = df_weekly.reset_index()
    
    return df_weekly


print("BẮT ĐẦU CHUYỂN ĐỔI SANG DỮ LIỆU TUẦN...")

for file_name in os.listdir(daily_weather_path):
    if file_name.endswith(".csv") and file_name.startswith("clean_"):
        df_weekly = resample_to_weekly(file_name)
        save_name = file_name.replace("clean_", "weekly_")
        save_path = os.path.join(weekly_weather_path, save_name)
        
        # Lưu ra CSV
        df_weekly.to_csv(save_path, index=False)
        print(f">> Đã chuyển sang data tuần và lưu: {save_name}")

print("\nHOÀN THÀNH ĐỒNG BỘ MỨC ĐỘ CHI TIẾT (WEEKLY)!")

