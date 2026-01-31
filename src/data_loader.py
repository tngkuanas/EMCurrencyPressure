
import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import datetime
import numpy as np
import os

# Define Constants
START_DATE = "2000-01-01"
END_DATE = datetime.datetime.today().strftime('%Y-%m-%d')
OUTPUT_FILE = "d:/DSProject/EM_Currency_Pressure/country_data_raw.csv"

# Country Mapping
COUNTRIES = {
    'Malaysia': {'iso2': 'MY', 'iso3': 'MYS', 'currency': 'MYR=X', 
                 'fred_cpi': 'MYRCPIALLMINMEI', 'fred_m2': 'MYAGM2MYM189N', 'fred_res': 'TRESEGMYM052N'},
    'Indonesia': {'iso2': 'ID', 'iso3': 'IDN', 'currency': 'IDR=X',
                  'fred_cpi': 'IDNCPIALLMINMEI', 'fred_m2': 'MYAGM2IDM189N', 'fred_res': 'TRESEGIDM052N'},
    'Thailand': {'iso2': 'TH', 'iso3': 'THA', 'currency': 'THB=X',
                 'fred_cpi': 'THACPIALLMINMEI', 'fred_m2': 'MYAGM2THM189N', 'fred_res': 'TRESEGTHM052N'},
    'Philippines': {'iso2': 'PH', 'iso3': 'PHL', 'currency': 'PHP=X',
                    'fred_cpi': 'PHLCPIALLMINMEI', 'fred_m2': 'MYAGM2PHM189N', 'fred_res': 'TRESEGPHM052N'},
    'India': {'iso2': 'IN', 'iso3': 'IND', 'currency': 'INR=X',
              'fred_cpi': 'INDCPIALLMINMEI', 'fred_m2': 'MYAGM2INM189N', 'fred_res': 'TRESEGINDM052N'}, # Check codes
    'Mexico': {'iso2': 'MX', 'iso3': 'MEX', 'currency': 'MXN=X',
               'fred_cpi': 'MEXCPIALLMINMEI', 'fred_m2': 'MYAGM2MXM189N', 'fred_res': 'TRESEGMXM052N'},
    'Brazil': {'iso2': 'BR', 'iso3': 'BRA', 'currency': 'BRL=X',
               'fred_cpi': 'BRACPIALLMINMEI', 'fred_m2': 'MYAGM2BRM189N', 'fred_res': 'TRESEGBRM052N'},
}
# Note: Codes are "best guess" based on standard OECD/IMF naming in FRED. 
# Some might fail. We will verify.

def fetch_metals():
    print("Fetching Metals data...")
    tickers = {
        'GC=F': 'Gold',
        'SI=F': 'Silver',
        'HG=F': 'Copper'
    }
    # Download
    try:
        df = yf.download(list(tickers.keys()), start=START_DATE, end=END_DATE, interval='1mo', progress=False)
        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            # We want Adj Close
            try:
                data = df['Adj Close'].copy()
            except KeyError:
                data = df['Close'].copy()
        else:
            data = df['Adj Close'] if 'Adj Close' in df else df['Close']
            
        data.rename(columns=tickers, inplace=True)
        # Resample to Month End to be safe and set index
        data.index = pd.to_datetime(data.index) + pd.offsets.MonthEnd(0)
        return data
    except Exception as e:
        print(f"Error fetching metals: {e}")
        return pd.DataFrame()

def fetch_fx_rates():
    print("Fetching FX data...")
    tickers = {v['currency']: k for k, v in COUNTRIES.items()}
    try:
        df = yf.download(list(tickers.keys()), start=START_DATE, end=END_DATE, interval='1mo', progress=False)
        
        if isinstance(df.columns, pd.MultiIndex):
            try:
                data = df['Adj Close'].copy()
            except KeyError:
                data = df['Close'].copy()
        else:
            data = df['Adj Close'] if 'Adj Close' in df else df['Close']
            
        data.rename(columns=tickers, inplace=True)
        data.index = pd.to_datetime(data.index) + pd.offsets.MonthEnd(0)
        return data
    except Exception as e:
        print(f"Error fetching FX: {e}")
        return pd.DataFrame()

def fetch_macro_for_country(country_name, start_date, end_date):
    c = COUNTRIES[country_name]
    # Define codes
    codes = {
        'CPI': c['fred_cpi'],
        'M2': c['fred_m2'],
        'Reserves': c['fred_res']
    }
    
    # Try fetching
    try:
        # Check if codes are valid (basic check)? No, just try fetch.
        df = web.DataReader(list(codes.values()), 'fred', start_date, end_date)
        
        # Rename columns back to keys
        # The result columns are the values (codes).
        reverse_map = {v: k for k, v in codes.items()}
        df.rename(columns=reverse_map, inplace=True)
        
        df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
        return df
    except Exception as e:
        print(f"Error fetching macro for {country_name} with codes {codes}: {e}")
        # Return empty DF or try partial?
        # For now, return empty but let's try to fetch individual if batch fails?
        # DataReader usually fetches what it can? No, it might fail all.
        return pd.DataFrame()

def main():
    if not os.path.exists(os.path.dirname(OUTPUT_FILE)):
        os.makedirs(os.path.dirname(OUTPUT_FILE))

    metals = fetch_metals()
    if metals.empty:
        print("Critical Error: No metals data.")
        return

    fx = fetch_fx_rates()
    if fx.empty:
        print("Critical Error: No FX data.")
        return
    
    all_data = []
    
    for country in COUNTRIES:
        print(f"Processing {country}...")
        
        # 1. Get Macro
        macro = fetch_macro_for_country(country, START_DATE, END_DATE)
        
        # 2. Get Country FX
        if country in fx.columns:
            country_fx = fx[[country]].copy()
            country_fx.rename(columns={country: 'FX_Rate'}, inplace=True)
        else:
            print(f"Warning: No FX data for {country}")
            country_fx = pd.DataFrame()

        # 3. Align everything
        # Base index should be the union of dates, or intersection?
        # We want meaningful data, so intersection of (Metals & FX) at least. 
        # Macro might be sparser or shorter.
        
        # Let's create a date range
        dates = pd.date_range(start=START_DATE, end=END_DATE, freq='M')
        base_df = pd.DataFrame(index=dates)
        
        # Merge Metals
        base_df = base_df.join(metals)
        
        # Merge FX
        if not country_fx.empty:
            base_df = base_df.join(country_fx)
            
        # Merge Macro
        if not macro.empty:
            base_df = base_df.join(macro)
        
        base_df['Country'] = country
        all_data.append(base_df)

    if all_data:
        final_df = pd.concat(all_data)
        # Drop rows where everything is null (except Country)
        cols_to_check = [c for c in final_df.columns if c != 'Country']
        final_df.dropna(how='all', subset=cols_to_check, inplace=True)
        
        final_df.to_csv(OUTPUT_FILE)
        print(f"Saved {len(final_df)} rows to {OUTPUT_FILE}")
        
        # S3 Simulation (Placeholder)
        try:
            import boto3
            # s3 = boto3.client('s3')
            # s3.upload_file(OUTPUT_FILE, 'my-em-data-bucket', 'country_data_raw.csv')
            print("Action: Upload to S3 (Simulated)")
        except ImportError:
            pass
            
        print(final_df.groupby('Country').count())
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()
