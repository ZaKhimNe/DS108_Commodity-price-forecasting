import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import json
import os

cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

def load_config():
    with open('config/coordinates.json', 'r') as file:
        return json.load(file)

def fetch_weather_for_location(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "et0_fao_evapotranspiration", "vapor_pressure_deficit_max"],
        "timezone": "auto"
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    daily = response.Daily()
    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    
    daily_data["temperature_2m_max"] = daily.Variables(0).ValuesAsNumpy()
    daily_data["temperature_2m_min"] = daily.Variables(1).ValuesAsNumpy()
    daily_data["precipitation_sum"] = daily.Variables(2).ValuesAsNumpy()
    daily_data["et0_fao_evapotranspiration"] = daily.Variables(3).ValuesAsNumpy()
    daily_data["vpd_max"] = daily.Variables(4).ValuesAsNumpy()
    
    return pd.DataFrame(data = daily_data)  

if __name__ == "__main__":
    config = load_config()
    START_DATE = "2021-01-01"
    END_DATE = "2025-12-31"
    
    os.makedirs('data/raw/weather', exist_ok=True)
    
    for commodity, details in config['commodities'].items():
        print(f"--- Đang tải thời tiết cho {commodity.upper()} ---")
        for loc in details['locations']:
            print(f"Kéo dữ liệu tọa độ: {loc['name']} ({loc['lat']}, {loc['lon']})")
            df_weather = fetch_weather_for_location(loc['lat'], loc['lon'], START_DATE, END_DATE)
            
            output_path = f"data/raw/weather/{commodity}_{loc['name']}.csv"
            df_weather.to_csv(output_path, index=False)
            print(f"-> Đã lưu {output_path}")