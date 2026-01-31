
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'em_currency_pressure_pipeline',
    default_args=default_args,
    description='Pipeline for EM Currency Pressure Analysis',
    schedule_interval=timedelta(days=30),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['em', 'finance'],
) as dag:

    # Define tasks
    # In a real setup, these would be PythonOperators or KubernetesPodOperators
    # For now, we simulate with Bash commands assuming scripts are in place
    
    t1_load_data = BashOperator(
        task_id='load_data',
        bash_command='python /opt/airflow/dags/src/data_loader.py',
    )

    t2_preprocess = BashOperator(
        task_id='preprocess',
        bash_command='python /opt/airflow/dags/src/preprocessor.py',
    )

    t3_train = BashOperator(
        task_id='train_model',
        bash_command='python /opt/airflow/dags/src/model_trainer.py',
    )

    # Define dependencies
    t1_load_data >> t2_preprocess >> t3_train
