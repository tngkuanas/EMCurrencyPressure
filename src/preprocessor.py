
import pandas as pd
import numpy as np
import os

INPUT_FILE = "d:/DSProject/EM_Currency_Pressure/country_data_raw.csv"
OUTPUT_FILE = "d:/DSProject/EM_Currency_Pressure/country_data_features.csv"

def load_data():
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    return df

def clean_data(df):
    # Ensure index is named Date for clarity
    df.index.name = 'Date'
    df = df.reset_index()
    # Sort by Country and Date
    df.sort_values(by=['Country', 'Date'], inplace=True)
    df.set_index('Date', inplace=True)
    
    # Group by country to avoid bleeding data between countries
    # Interpolate limit direction both? Or just ffill.
    # Financial data: often ffill is safer. Macro: Interpolate time.
    
    cleaned_dfs = []
    for country, group in df.groupby('Country'):
        group = group.copy()
        
        # 1. Fill Metals (Global, should be same, but just in case)
        # Metals columns: Gold, Silver, Copper
        for col in ['Gold', 'Silver', 'Copper']:
            group[col] = group[col].interpolate(method='linear').ffill().bfill()
            
        # 2. Fill FX
        # If FX is missing, maybe ffill
        if 'FX_Rate' in group.columns:
            group['FX_Rate'] = group['FX_Rate'].interpolate(method='linear').ffill().bfill()
            
        # 3. Fill Macro
        # CPI, M2, Reserves might be sparse. Interpolate time.
        for col in ['CPI', 'M2', 'Reserves']:
            if col in group.columns:
                group[col] = group[col].interpolate(method='time').ffill().bfill()
        
        cleaned_dfs.append(group)
        
    return pd.concat(cleaned_dfs)

def compute_features(df):
    dfs = []
    for country, group in df.groupby('Country'):
        group = group.sort_index().copy()
        
        # --- Metals Features ---
        for metal in ['Gold', 'Silver', 'Copper']:
            # Rolling Metrics
            group[f'{metal}_3m_chg'] = group[metal].pct_change(3)
            group[f'{metal}_12m_chg'] = group[metal].pct_change(12)
            
            # Spike Detection
            roll_mean = group[metal].rolling(window=12).mean()
            roll_std = group[metal].rolling(window=12).std()
            z_score = (group[metal] - roll_mean) / (roll_std + 1e-9) # Avoid div by zero
            group[f'{metal}_Z'] = z_score
            group[f'{metal}_Spike'] = (z_score > 1.5).astype(int)
            
        # --- FX Features ---
        if 'FX_Rate' in group.columns:
            # FX Returns (Depreciation = Positive if quote is USD/LCY? No, usually LCY/USD e.g. MYR 4.5)
            # If Quote is MYR=X (MYR per USD), then Increase = Depreciation of MYR.
            # So Positive Change = Pressure.
            
            group['FX_1m_chg'] = group['FX_Rate'].pct_change(1)
            group['FX_3m_chg'] = group['FX_Rate'].pct_change(3)
            
            # Volatility
            group['FX_Vol_3m'] = group['FX_1m_chg'].rolling(window=3).std()
            
            # Pressure Index components
            # Pressure = Deviation from 12m moving average?
            fx_ma_12 = group['FX_Rate'].rolling(window=12).mean()
            group['FX_Dev_MA12'] = (group['FX_Rate'] - fx_ma_12) / fx_ma_12
            
        # --- Macro Features ---
        if 'CPI' in group.columns:
            # Inflation (YoY)
            group['Inflation_YoY'] = group['CPI'].pct_change(12)
        if 'M2' in group.columns:
            group['M2_Growth_12m'] = group['M2'].pct_change(12)
            group['M2_Growth_3m'] = group['M2'].pct_change(3)
        if 'Reserves' in group.columns:
            group['Reserves_Chg_12m'] = group['Reserves'].pct_change(12)

        dfs.append(group)
    
    return pd.concat(dfs)

def define_targets(df):
    # Definition:
    # "Pressure = 1" if currency shows top 20% worst FX movement combined with inflation spike?
    # Let's simplify: 
    # Continuous Risk Score = FX_Depreciation_Score + Inflation_Score + Metal_Drop_Score?
    # User said: "Pressure = 1 if currency shows top 20% worst FX movement combined with inflation spike"
    
    # We'll compute percentiles per country or global?
    # Let's do Global percentiles for "EM comparison".
    
    # 1. FX Pressure Metric: Monthly Depreciation
    # Handle NaN
    df['FX_1m_chg'] = df['FX_1m_chg'].fillna(0)
    
    # High depreciation = High Pressure.
    # We define "Crisis" as FX_1m_chg > 95th percentile? OR > 20% worst.
    
    # Let's use a Risk Score (Continuous) first.
    # Risk Score = (FX_Z + Inflation_Z - Reserves_Z)/3  (Reserves drop is bad)
    # We need to normalize these features first.
    
    cols_to_norm = ['FX_1m_chg', 'Inflation_YoY', 'Reserves_Chg_12m']
    # Check if cols exist
    available_cols = [c for c in cols_to_norm if c in df.columns]
    
    for col in available_cols:
        # Z-score normalization
        mean = df[col].mean()
        std = df[col].std()
        df[f'{col}_Z_Global'] = (df[col] - mean) / (std + 1e-9)
    
    # Simple Composite Risk Score
    # Risk increases if FX depreciation is high (+), Inflation is high (+), Reserves dropping (-)
    # Reserves_Chg_12m Z is positive if reserves grew. So we subtract it.
    
    risk_score = 0
    if 'FX_1m_chg_Z_Global' in df.columns:
        risk_score += df['FX_1m_chg_Z_Global']
    if 'Inflation_YoY_Z_Global' in df.columns:
        risk_score += df['Inflation_YoY_Z_Global']
    if 'Reserves_Chg_12m_Z_Global' in df.columns:
        risk_score -= df['Reserves_Chg_12m_Z_Global'] # Lower reserves = Higher Risk
        
    df['Risk_Score'] = risk_score
    
    # Target Label for Classification
    # Top 20% of Risk Score = 1, else 0.
    threshold = df['Risk_Score'].quantile(0.80)
    df['Target_Pressure'] = (df['Risk_Score'] > threshold).astype(int)
    
    # Shift Target? 
    # Usually we want to PREDICT pressure in t+1 using features at t.
    # "Input: feature vectors per month" -> "Output: risk label". 
    # Are we detecting simultaneous risk or predicting future risk?
    # "Early-warning" implies prediction. 
    # So we should Shift the Target backwards by 1 month (Predict t+1 using t).
    
    df['Target_Next_Month_Pressure'] = df.groupby('Country')['Target_Pressure'].shift(-1)
    df['Target_Next_Month_Risk'] = df.groupby('Country')['Risk_Score'].shift(-1)
    
    # Drop rows where Target is NaN (last month)
    df.dropna(subset=['Target_Next_Month_Pressure'], inplace=True)
    
    return df

def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return
        
    df = load_data()
    print(f"Loaded {len(df)} rows.")
    
    df = clean_data(df)
    print("Data cleaned.")
    
    df = compute_features(df)
    print("Features computed.")
    
    df = define_targets(df)
    print("Targets defined.")
    
    # Save
    df.to_csv(OUTPUT_FILE)
    print(f"Saved features to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
