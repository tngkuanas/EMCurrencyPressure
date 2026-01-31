
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, recall_score, mean_squared_error, r2_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

import mlflow
import mlflow.xgboost

INPUT_FILE = "d:/DSProject/EM_Currency_Pressure/country_data_features.csv"
OUTPUT_DIR = "d:/DSProject/EM_Currency_Pressure/results"
PLOTS_DIR = "d:/DSProject/EM_Currency_Pressure/plots"

# MLflow Setup
mlflow.set_tracking_uri("file:///d:/DSProject/EM_Currency_Pressure/mlruns")
mlflow.set_experiment("EM_Currency_Pressure")

SPLIT_DATE = "2019-01-01"

FEATURES = [
    'Gold_3m_chg', 'Silver_3m_chg', 'Copper_3m_chg',
    'Gold_12m_chg', 'Copper_12m_chg',
    'Gold_Z', 'Copper_Z', 'Gold_Spike', 'Copper_Spike',
    'FX_Vol_3m', 'FX_Dev_MA12',
    'Inflation_YoY', 'M2_Growth_12m', 'Reserves_Chg_12m'
]

def load_data():
    df = pd.read_csv(INPUT_FILE, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    # Replace infinite values with NaN and then fill or clip
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True) # Fill remaining NaNs with 0
    return df

def train_and_evaluate(df):
    # Ensure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Check features exist
    available_features = [f for f in FEATURES if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Split
    train = df[df.index < SPLIT_DATE]
    test = df[df.index >= SPLIT_DATE]
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    
    # --- Classification ---
    print("\nTraining Classifier (Pressure Event)...")
    target_cls = 'Target_Next_Month_Pressure'
    
    X_train = train[available_features]
    y_train_cls = train[target_cls]
    X_test = test[available_features]
    y_test_cls = test[target_cls]
    
    # MLflow Autolog or Manual
    mlflow.xgboost.autolog()
    
    with mlflow.start_run(run_name="XGB_Classifier"):
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        clf.fit(X_train, y_train_cls)
        
        y_pred_cls = clf.predict(X_test)
        y_prob_cls = clf.predict_proba(X_test)[:, 1]
        
        roc = roc_auc_score(y_test_cls, y_prob_cls)
        prec = precision_score(y_test_cls, y_pred_cls, zero_division=0)
        rec = recall_score(y_test_cls, y_pred_cls)
        
        mlflow.log_metrics({"roc_auc": roc, "precision": prec, "recall": rec})
        print(f"Logged Classifier metrics to MLflow")
    
    print(f"ROC-AUC: {roc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(classification_report(y_test_cls, y_pred_cls))
    
    # Feature Importance (Classifier)
    imp_cls = pd.Series(clf.feature_importances_, index=available_features).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=imp_cls.values, y=imp_cls.index)
    plt.title('Feature Importance (Classification)')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/feature_importance_classifier.png')
    plt.close()

    # --- Regression ---
    print("\nTraining Regressor (Risk Score)...")
    target_reg = 'Target_Next_Month_Risk'
    
    X_train = train[available_features]
    y_train_reg = train[target_reg]
    X_test = test[available_features]
    y_test_reg = test[target_reg]
    
    reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    reg.fit(X_train, y_train_reg)
    
    y_pred_reg = reg.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    
    # Plot Prediction vs Actual (for Test set subset, e.g. Malaysia)
    test_with_pred = test.copy()
    test_with_pred['Predicted_Risk'] = y_pred_reg
    
    # Save Predictions
    test_with_pred[['Country', target_cls, target_reg, 'Predicted_Risk']].to_csv(f'{OUTPUT_DIR}/predictions.csv')
    
    # Plot for Malaysia
    country = 'Malaysia'
    country_data = test_with_pred[test_with_pred['Country'] == country]
    
    if not country_data.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(country_data.index, country_data[target_reg], label='Actual Risk', alpha=0.7)
        plt.plot(country_data.index, country_data['Predicted_Risk'], label='Predicted Risk', linestyle='--')
        plt.title(f'{country}: Actual vs Predicted Risk Score (Test Set)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/malaysia_risk_prediction.png')
        plt.close()
        
    # Plot for Peers (Aggregated)
    # Average Risk Score over time
    avg_risk = test_with_pred.groupby('Date')['Predicted_Risk'].mean()
    plt.figure(figsize=(12, 6))
    plt.plot(avg_risk.index, avg_risk.values, label='Average EM Risk Score')
    if not country_data.empty:
        plt.plot(country_data.index, country_data['Predicted_Risk'], label='Malaysia Risk Score')
    plt.title('EM Average Risk Score vs Malaysia')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/peer_comparison_risk.png')
    plt.close()

    # --- Save Metrics ---
    metrics = {
        'Classifier_ROC_AUC': roc,
        'Classifier_Precision': prec,
        'Classifier_Recall': rec,
        'Regressor_RMSE': rmse,
        'Regressor_R2': r2
    }
    pd.DataFrame([metrics]).to_csv(f'{OUTPUT_DIR}/metrics.csv', index=False)
    print(f"\nMetrics saved to {OUTPUT_DIR}/metrics.csv")

    # --- Save Models ---
    MODELS_DIR = "d:/DSProject/EM_Currency_Pressure/models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    clf.save_model(f'{MODELS_DIR}/classifier.json')
    reg.save_model(f'{MODELS_DIR}/regressor.json')
    print(f"Models saved to {MODELS_DIR}")

def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return
        
    df = load_data()
    print(f"Loaded {len(df)} rows.")
    
    train_and_evaluate(df)

if __name__ == "__main__":
    main()
