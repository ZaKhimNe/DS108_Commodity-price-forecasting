import yfinance as yf
import pandas as pd
import json
import os

def load_config():
    with open('config/coordinates.json', 'r') as file:
        return json.load(file)
    
def fetch_market_data(ticker, start_date, end_date, output_name):
    print(f"Đang tải dữ liệu thị trường cho {ticker}...")

    df = yf.download(ticker, start = start_date, end = end_date)

    if df.empty:
        print(f"Lỗi không tìm thấy dữ liệu cho {ticker}")
        return
    
    df.reset_index(inplace=True)

    os.makedirs('data/raw/market', exist_ok=True)
    output_path = f"data/raw/market/{output_name}_market.csv"
    df.to_csv(output_path, index=False)

    print(f"Thành công! Đã lưu: {output_path}")

if __name__ == "__main__":
    config = load_config()

    START_DATE = "2021-01-01"
    END_DATE = "2026-01-01"

    fetch_market_data(config['commodities']['corn']['ticker'], START_DATE, END_DATE, "corn")
    fetch_market_data(config['commodities']['coffee']['ticker'], START_DATE, END_DATE, "coffee")
