
# SovereignSentinel: Autonomous EM Currency Risk Infrastructure

**An autonomous Early Warning System that leverages Spark-driven data engineering and XGBoost models to predict emerging market currency depreciation events through a containerized, production-grade MLOps pipeline.**

## 🚀 Overview
SovereignSentinel monitors hidden stress signals in global commodity prices and national economic data to predict currency crashes or major sell-offs in emerging markets (EM) with a 30-day lead time.

## 🛠️ Tech Stack
- **Languages**: Python (Pandas, NumPy, Scikit-Learn)
- **Big Data**: PySpark (Window Functions, Distributed Processing)
- **Modeling**: XGBoost (Classifier & Regressor)
- **MLOps**: MLflow (Experiment Tracking), Apache Airflow (Orchestration)
- **DevOps**: Docker, Kubernetes (K8s), Prometheus, Grafana
- **Data**: AWS S3, Yahoo Finance, FRED API

## 🏛️ Architecture
1. **Ingestion**: Automated recovery of metals (Gold, Copper) and Macro (CPI, M2, Reserves) data.
2. **Processing**: Spark-based feature engineering (Rolling Volatility, Z-Scores, Debasement Metrics).
3. **Training**: XGBoost models tracked via MLflow with automated retraining cycles.
4. **Serving**: High-performance FastAPI endpoint containerized with Docker.
5. **Observability**: Prometheus instrumentation for real-time monitoring of model drift and system health.

## 🚦 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (Data -> Spark -> Training)
python src/pipeline.py

# Start the API
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## 📊 MLOps Vision
This project is designed as a **Cloud-Agnostic** MRE (Modular Runtime Environment). It can be deployed across AWS, GCP, or Azure using the provided Kubernetes manifests and Airflow DAGs.
