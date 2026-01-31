
from fastapi import FastAPI, HTTPException
import pandas as pd
import xgboost as xgb
import os
import json
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize App
app = FastAPI(title="EM Currency Pressure API", version="1.0")

# Instrument Prometheus
Instrumentator().instrument(app).expose(app)

# Paths
MODELS_DIR = "d:/DSProject/EM_Currency_Pressure/models"
FEATURE_FILE = "d:/DSProject/EM_Currency_Pressure/country_data_features.csv"

# Load Models
classifier = xgb.XGBClassifier()
regressor = xgb.XGBRegressor()

try:
    classifier.load_model(f"{MODELS_DIR}/classifier.json")
    regressor.load_model(f"{MODELS_DIR}/regressor.json")
    MODELS_LOADED = True
except Exception as e:
    print(f"Error loading models: {e}")
    MODELS_LOADED = False

# Load Data (for 'latest' endpoint)
def get_latest_data():
    if os.path.exists(FEATURE_FILE):
        df = pd.read_csv(FEATURE_FILE)
        # Assuming last row per country is 'latest' if sorted
        # Or filter by max Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            latest_date = df['Date'].max()
            latest_df = df[df['Date'] == latest_date]
            return latest_df
    return pd.DataFrame()

@app.get("/")
def home():
    return {"message": "EM Currency Pressure API is running", "models_loaded": MODELS_LOADED}

@app.get("/latest_risks")
def get_latest_risks():
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    df = get_latest_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Features required
    # Use same list as training (should be configurable)
    FEATURES = [
        'Gold_3m_chg', 'Silver_3m_chg', 'Copper_3m_chg',
        'Gold_12m_chg', 'Copper_12m_chg',
        'Gold_Z', 'Copper_Z', 'Gold_Spike', 'Copper_Spike',
        'FX_Vol_3m', 'FX_Dev_MA12',
        'Inflation_YoY', 'M2_Growth_12m', 'Reserves_Chg_12m'
    ]
    
    # Handle missing cols with 0
    X = df[FEATURES].fillna(0)
    
    # Predict
    risk_scores = regressor.predict(X)
    pressure_probs = classifier.predict_proba(X)[:, 1]
    
    results = []
    for i, row in df.reset_index().iterrows():
        results.append({
            "Country": row['Country'],
            "Date": str(row['Date'].date()),
            "Risk_Score": float(risk_scores[i]),
            "Pressure_Probability": float(pressure_probs[i])
        })
        
    return {"date": str(df['Date'].iloc[0].date()), "risks": results}

@app.post("/predict_manual")
def predict_manual(features: dict):
    # features: dict of feature_name -> value
    # Validate input?
    if not MODELS_LOADED:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Ensure all columns exist
    FEATURES = [
        'Gold_3m_chg', 'Silver_3m_chg', 'Copper_3m_chg',
        'Gold_12m_chg', 'Copper_12m_chg',
        'Gold_Z', 'Copper_Z', 'Gold_Spike', 'Copper_Spike',
        'FX_Vol_3m', 'FX_Dev_MA12',
        'Inflation_YoY', 'M2_Growth_12m', 'Reserves_Chg_12m'
    ]
    
    for f in FEATURES:
        if f not in df.columns:
            df[f] = 0.0
            
    risk = regressor.predict(df[FEATURES])[0]
    prob = classifier.predict_proba(df[FEATURES])[:, 1][0]
    
    return {"Risk_Score": float(risk), "Pressure_Probability": float(prob)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
