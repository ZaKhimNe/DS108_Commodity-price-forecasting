import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "farming")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed", "farming")

# Tạo thư mục đầu ra nếu chưa tồn tại
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_calendar(file_name, output_name):
    print(f"Đang xử lý: {file_name}")
    
    input_path = os.path.join(INPUT_DIR, file_name)
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    df = pd.read_csv(input_path)
    
    # Tích hợp xử lý timezone utc=True (Từ file thứ 2)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.set_index('Date')

    # Thay đổi tần suất thời gian sang tuần và lấp khuyết
    stage_cols = [col for col in df.columns if col.startswith('is_')]
    weekly_df = df[stage_cols].resample('W').max().fillna(0)

    # Tính toán thời lượng (duration) các giai đoạn sinh trưởng
    for col in stage_cols:
        weekly_df[f'{col}_duration'] = weekly_df[col].groupby(
            (weekly_df[col] != weekly_df[col].shift()).cumsum()
        ).cumsum()

    # Tích hợp mã hóa chu kỳ thời gian - Cyclical Encoding (Từ file thứ 1)
    weekly_df['month'] = weekly_df.index.month
    weekly_df['week_of_year'] = weekly_df.index.isocalendar().week

    weekly_df['sin_month'] = np.sin(2 * np.pi * weekly_df['month'] / 12)
    weekly_df['cos_month'] = np.cos(2 * np.pi * weekly_df['month'] / 12)
    weekly_df['sin_week'] = np.sin(2 * np.pi * weekly_df['week_of_year'] / 52)
    weekly_df['cos_week'] = np.cos(2 * np.pi * weekly_df['week_of_year'] / 52)

    # Lưu kết quả
    weekly_df.to_csv(output_path)
    print(f"-> Hoàn tất lưu tại: {output_path}")
    
    return weekly_df

# Thực thi
if __name__ == "__main__":
    coffee_cal_weekly = process_calendar('coffee_calendar.csv', 'weekly_coffee_calendar.csv')
    corn_cal_weekly = process_calendar('corn_calendar.csv', 'weekly_corn_calendar.csv')