import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "farming")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "preprocessed", "farming")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def process_calendar(file_name, output_name):
    print(f"Đang xử lý: {file_name}")
    df = pd.read_csv(os.path.join(INPUT_DIR, file_name))
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.set_index('Date')

    stage_cols = [col for col in df.columns if col.startswith('is_')]
    weekly_df = df[stage_cols].resample('W').max().fillna(0)

    for col in stage_cols:
        weekly_df[f'{col}_duration'] = weekly_df[col].groupby((weekly_df[col] != weekly_df[col].shift()).cumsum()).cumsum()

    weekly_df.to_csv(os.path.join(OUTPUT_DIR, output_name))
    print(f"-> Hoàn tất lưu tại: {output_name}")
    return weekly_df

# Thực thi
coffee_cal_weekly = process_calendar('coffee_calendar.csv', 'weekly_coffee_calendar.csv')
corn_cal_weekly = process_calendar('corn_calendar.csv', 'weekly_corn_calendar.csv')