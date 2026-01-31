
import os
import subprocess
import datetime

# Define step scripts
SCRIPTS_DIR = "d:/DSProject/EM_Currency_Pressure/src"

STEPS = [
    ("Data Loading", f"python {SCRIPTS_DIR}/data_loader.py"),
    ("Preprocessing", f"python {SCRIPTS_DIR}/preprocessor.py"),
    ("Training", f"python {SCRIPTS_DIR}/model_trainer.py"),
    ("Spark Bench (Proof of Concept)", f"python {SCRIPTS_DIR}/spark_processor.py")
]

def run_pipeline():
    log_file = f"d:/DSProject/EM_Currency_Pressure/pipeline_log.txt"
    
    with open(log_file, "a") as f:
        f.write(f"\n--- Pipeline Started: {datetime.datetime.now()} ---\n")
        
    print("Starting Pipeline...")
    
    for name, cmd in STEPS:
        print(f"Running {name}...")
        try:
            result = subprocess.run(cmd, check=True, shell=True, capture_output=True, text=True)
            with open(log_file, "a") as f:
                f.write(f"[{datetime.datetime.now()}] {name} SUCCESS\n")
        except subprocess.CalledProcessError as e:
            err_msg = f"[{datetime.datetime.now()}] {name} FAILED: {e.stderr}"
            print(err_msg)
            with open(log_file, "a") as f:
                f.write(err_msg + "\n")
            return 
            
    print("Pipeline Completed Successfully.")

if __name__ == "__main__":
    run_pipeline()
