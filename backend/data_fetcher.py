import pandas as pd
import time
import os
from vnstock import Vnstock
from datetime import datetime, timedelta

def get_vn100_symbols():
    # A representative subset of VN100 for testing purposes
    # Using 10 major symbols to keep it manageable but realistic
    return ["FPT", "HPG", "VNM", "VCB", "VIC", "SSI", "MWG", "VPB", "TCB", "MBB"]

def fetch_data_for_symbols(symbols, start_date, end_date, output_dir="backend/data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_data = []

    print(f"Fetching data from {start_date} to {end_date} for {len(symbols)} symbols...")

    for i, symbol in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] Fetching {symbol}...")

        try:
            # Use vnstock with source KBS to avoid 403 Forbidden issues seen with VCI
            stock = Vnstock().stock(symbol=symbol, source='KBS')
            df = stock.quote.history(start=start_date, end=end_date)

            if df is not None and not df.empty:
                df['ticker'] = symbol

                # Format time column to extract just the date if it has time
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')

                # Save individual symbol data
                df.to_csv(f"{output_dir}/{symbol}.csv", index=False)
                all_data.append(df)
            else:
                print(f"No data returned for {symbol}")

        except Exception as e:
            print(f"Error fetching {symbol}: {e}")

        print("Sleeping for 3.2 seconds to respect rate limits...")
        time.sleep(3.2)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(f"{output_dir}/vn100_subset.csv", index=False)
        print(f"Combined data saved to {output_dir}/vn100_subset.csv")
        return combined_df
    else:
        print("No data was fetched.")
        return None

if __name__ == "__main__":
    symbols = get_vn100_symbols()

    # We need at least ~1000 days of data for Walk-Forward testing and long MAs (e.g. 200, 252)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=1500)).strftime("%Y-%m-%d")

    fetch_data_for_symbols(symbols, start_date, end_date)
