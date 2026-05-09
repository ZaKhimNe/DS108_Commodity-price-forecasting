import yfinance as yf
import pandas as pd
import os
import requests
import json

def fetch_exchange_rate(ticker, start_date, end_date, output_name):
    print(f"Đang tải tỷ giá {ticker}...")
    df = yf.download(ticker, start=start_date, end=end_date)
    if not df.empty:
        df.reset_index(inplace=True)
        os.makedirs('data/raw/macro', exist_ok=True)
        df.to_csv(f"data/raw/macro/{output_name}_exchange.csv", index=False)
        print(f"-> Đã lưu tỷ giá {output_name}")
    else:
        print(f"Lỗi: Không tải được tỷ giá {ticker}")

def fetch_inflation_bls(start_year, end_year):
    print("Đang tải dữ liệu lạm phát (CPI) từ nguồn gốc (BLS - Mỹ)...")
    
    headers = {'Content-type': 'application/json'}
    data = json.dumps({
        "seriesid": ['CUUR0000SA0'],
        "startyear": str(start_year),
        "endyear": str(end_year)
    })
    
    try:
        response = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers, timeout=15)
        response.raise_for_status()
        json_data = json.loads(response.text)
        
        if json_data['status'] == 'REQUEST_SUCCEEDED':
            records = []
            
            for series in json_data['Results']['series']:
                for item in series['data']:
                    year = item['year']
                    period = item['period'] 
                    value = item['value']
                    
                    if period != 'M13': 
                        month = period.replace('M', '')
                        date_str = f"{year}-{month}-01"
                        
                        # XỬ LÝ LỖI DỮ LIỆU BẨN TẠI ĐÂY
                        try:
                            # Cố gắng ép thành số
                            numeric_value = float(value)
                            records.append({'Date': date_str, 'US_CPI': numeric_value})
                        except ValueError:
                            # Nếu ép kiểu thất bại (do chứa dấu '-'), in ra thông báo và bỏ qua tháng đó
                            print(f"  -> Bỏ qua dữ liệu tháng {month}/{year} vì chưa có số liệu chính thức (giá trị: '{value}')")
            
            cpi_df = pd.DataFrame(records)
            cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])
            cpi_df = cpi_df.sort_values('Date').reset_index(drop=True)
            
            os.makedirs('data/raw/macro', exist_ok=True)
            cpi_df.to_csv("data/raw/macro/us_inflation.csv", index=False)
            print("-> Đã lưu dữ liệu lạm phát (Dữ liệu thật 100% từ Cục Thống kê Mỹ)")
        else:
            print(f"Lỗi từ máy chủ BLS: {json_data.get('message', 'Không rõ')}")
            
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu BLS: {e}")

if __name__ == "__main__":
    START_DATE = "2021-01-01"
    END_DATE = "2026-01-01"
    
    # API của BLS yêu cầu định dạng năm riêng biệt
    START_YEAR = 2021
    END_YEAR = 2026
    
    # Lấy tỷ giá
    fetch_exchange_rate("BRL=X", START_DATE, END_DATE, "usd_brl")
    
    # Lấy lạm phát bằng API công khai của BLS
    fetch_inflation_bls(START_YEAR, END_YEAR)