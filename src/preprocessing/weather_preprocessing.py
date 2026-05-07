import pandas as pd
import os
from sklearn.impute import KNNImputer

raw_weather_path = "data/preprocessed/weather/weather_despiked"
processed_path = "data/preprocessed/weather/weather_clean"
os.makedirs(processed_path, exist_ok=True)

def preprocess_weather_file(file_name):    
    df = pd.read_csv(os.path.join(raw_weather_path, file_name))
    
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    df = df.sort_values('date').set_index('date')

    print(f"\n--- Missing values trong {file_name} ---")
    print(df.isnull().sum())

    weather_cols = ['temperature_2m_max', 'et0_fao_evapotranspiration']
    existing_cols = [col for col in weather_cols if col in df.columns]
    
    if existing_cols:
        df[existing_cols] = df[existing_cols].interpolate(method='spline', order=3, limit_direction='both')
        
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[existing_cols] = imputer.fit_transform(df[existing_cols])

    if 'precipitation_sum' in df.columns:
        df['precipitation_sum'] = df['precipitation_sum'].fillna(0)
    if 'temperature_2m_max' in df.columns:
        df['temp_max_rolling_7d'] = df['temperature_2m_max'].rolling(window=7).mean()
        df['temp_max_rolling_14d'] = df['temperature_2m_max'].rolling(window=14).mean()
    if 'precipitation_sum' in df.columns:
        df['precip_sum_rolling_7d'] = df['precipitation_sum'].rolling(window=7).sum()

    df = df.dropna()
    df = df.reset_index()
    return df

print("BẮT ĐẦU QUÉT THƯ MỤC...")

for file_name in os.listdir(raw_weather_path):
    if file_name.endswith(".csv"):
        cleaned_df = preprocess_weather_file(file_name)
        
        save_name = f"clean_{file_name}"
        save_path = os.path.join(processed_path, save_name)
        
        cleaned_df.to_csv(save_path, index=False)
        print(f">> Đã xử lý và lưu thành công: {save_name}")

print("\nHOÀN THÀNH TOÀN BỘ!")