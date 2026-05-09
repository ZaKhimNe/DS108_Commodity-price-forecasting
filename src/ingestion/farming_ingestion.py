import pandas as pd
import json
import os

def load_config():
    with open('config/coordinates.json', 'r') as file:
        return json.load(file)

def generate_crop_calendar(commodity_name, start_year, end_year, config_data):
    print(f"Đang khởi tạo lịch mùa vụ cho {commodity_name.upper()}...")
    
    # Tạo chuỗi thời gian liên tục từ ngày bắt đầu đến kết thúc
    dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='D')
    df = pd.DataFrame({'Date': dates})
    df['Month'] = df['Date'].dt.month
    
    cal_info = config_data['calendar'][commodity_name]
    
    if commodity_name == 'coffee':
        df['is_flowering'] = df['Month'].isin(cal_info['flowering_months']).astype(int)
        df['is_harvest'] = df['Month'].isin(cal_info['harvest_months']).astype(int)
    
    elif commodity_name == 'corn':
        df['is_planting'] = df['Month'].isin(cal_info['planting_months']).astype(int)
        df['is_pollination'] = df['Month'].isin(cal_info['pollination_months']).astype(int)
        df['is_harvest'] = df['Month'].isin(cal_info['harvest_months']).astype(int)
    
    df.drop(columns=['Month'], inplace=True)
    
    os.makedirs('data/raw/farming', exist_ok=True)
    df.to_csv(f"data/raw/farming/{commodity_name}_calendar.csv", index=False)
    print(f"-> Đã lưu lịch mùa vụ: data/raw/farming/{commodity_name}_calendar.csv")

if __name__ == "__main__":
    config = load_config()
    START_YEAR = 2021
    END_YEAR = 2026
    
    generate_crop_calendar('coffee', START_YEAR, END_YEAR, config)
    generate_crop_calendar('corn', START_YEAR, END_YEAR, config)