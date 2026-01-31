
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lag, avg, stddev, last, expr
from pyspark.sql.window import Window
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("EM Currency Pressure Spark Processor") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# Paths
INPUT_FILE = "d:/DSProject/EM_Currency_Pressure/country_data_raw.csv"
OUTPUT_FILE = "d:/DSProject/EM_Currency_Pressure/country_data_features_spark.csv"

def process_data_spark():
    print(f"Reading raw data from {INPUT_FILE}...")
    # Load data
    df = spark.read.csv(INPUT_FILE, header=True, inferSchema=True)
    print(f"Raw data schema loaded. Rows found: {df.count()}")
    
    # Rename 'index' to 'Date' if it came from the raw CSV
    if "index" in df.columns:
        df = df.withColumnRenamed("index", "Date")
    print(f"Columns: {df.columns}")
        
    # Cast Date
    df = df.withColumn("Date", col("Date").cast("date"))
    
    # Define Windows
    # Ordered by Date within each Country
    w_country = Window.partitionBy("Country").orderBy("Date")
    # Rolling windows (including current row)
    w_3m = w_country.rowsBetween(-2, 0) 
    w_12m = w_country.rowsBetween(-11, 0)
    
    print("Computing Spark Features...")
    
    # --- 1. Metals Features ---
    for metal in ["Gold", "Silver", "Copper"]:
        # Pct Change (approximate since it's monthly)
        # 3m change: (current - lag2) / lag2
        df = df.withColumn(f"{metal}_lag3", lag(col(metal), 3).over(w_country))
        df = df.withColumn(f"{metal}_3m_chg", (col(metal) - col(f"{metal}_lag3")) / col(f"{metal}_lag3"))
        
        df = df.withColumn(f"{metal}_lag12", lag(col(metal), 12).over(w_country))
        df = df.withColumn(f"{metal}_12m_chg", (col(metal) - col(f"{metal}_lag12")) / col(f"{metal}_lag12"))
        
        # Z-Scores and Spikes
        df = df.withColumn(f"{metal}_mean_12m", avg(col(metal)).over(w_12m))
        df = df.withColumn(f"{metal}_std_12m", stddev(col(metal)).over(w_12m))
        df = df.withColumn(f"{metal}_Z", (col(metal) - col(f"{metal}_mean_12m")) / (col(f"{metal}_std_12m") + 1e-9))
        df = df.withColumn(f"{metal}_Spike", when(col(f"{metal}_Z") > 1.5, 1).otherwise(0))

    # --- 2. FX Features ---
    # 1m change
    df = df.withColumn("FX_lag1", lag(col("FX_Rate"), 1).over(w_country))
    df = df.withColumn("FX_1m_chg", (col("FX_Rate") - col("FX_lag1")) / col("FX_lag1"))
    
    # 3m Volatility (std of 1m returns)
    df = df.withColumn("FX_Vol_3m", stddev(col("FX_1m_chg")).over(w_3m))
    
    # Deviation from MA12
    df = df.withColumn("FX_MA12", avg(col("FX_Rate")).over(w_12m))
    df = df.withColumn("FX_Dev_MA12", (col("FX_Rate") - col("FX_MA12")) / col("FX_MA12"))
    
    # --- 3. Macro Features ---
    # Inflation YoY
    df = df.withColumn("CPI_lag12", lag(col("CPI"), 12).over(w_country))
    df = df.withColumn("Inflation_YoY", (col("CPI") - col("CPI_lag12")) / col("CPI_lag12"))
    
    # M2 Growth
    df = df.withColumn("M2_lag12", lag(col("M2"), 12).over(w_country))
    df = df.withColumn("M2_Growth_12m", (col("M2") - col("M2_lag12")) / col("M2_lag12"))
    
    # Reserves Change
    df = df.withColumn("Reserves_lag12", lag(col("Reserves"), 12).over(w_country))
    df = df.withColumn("Reserves_Chg_12m", (col("Reserves") - col("Reserves_lag12")) / col("Reserves_lag12"))

    # --- 4. Target Definition (Global Stats) ---
    # For global stats, we need a Window across all countries for each date
    # Actually, the user wants a composite risk score based on Z-scores.
    
    # Normalizing columns globally (Z-score normalization)
    stats = df.select(
        avg("FX_1m_chg").alias("avg_fx"), stddev("FX_1m_chg").alias("std_fx"),
        avg("Inflation_YoY").alias("avg_inf"), stddev("Inflation_YoY").alias("std_inf"),
        avg("Reserves_Chg_12m").alias("avg_res"), stddev("Reserves_Chg_12m").alias("std_res")
    ).collect()[0]
    
    df = df.withColumn("FX_Z_Global", (col("FX_1m_chg") - stats['avg_fx']) / (stats['std_fx'] + 1e-9))
    df = df.withColumn("Inf_Z_Global", (col("Inflation_YoY") - stats['avg_inf']) / (stats['std_inf'] + 1e-9))
    df = df.withColumn("Res_Z_Global", (col("Reserves_Chg_12m") - stats['avg_res']) / (stats['std_res'] + 1e-9))
    
    df = df.withColumn("Risk_Score", col("FX_Z_Global") + col("Inf_Z_Global") - col("Res_Z_Global"))
    
    # Target Pressure (Top 20%)
    # In Spark, we can use percent_rank
    w_global = Window.orderBy("Risk_Score")
    df = df.withColumn("Risk_Percentile", expr("percent_rank()").over(w_global))
    df = df.withColumn("Target_Pressure", when(col("Risk_Percentile") > 0.8, 1).otherwise(0))
    
    # Look-ahead: Shift Target Next Month
    # We lead the risk score and target label
    df = df.withColumn("Target_Next_Month_Pressure", lag(col("Target_Pressure"), -1).over(w_country))
    df = df.withColumn("Target_Next_Month_Risk", lag(col("Risk_Score"), -1).over(w_country))
    
    print("Writing processed features to CSV...")
    # On Windows, Spark often fails to write to disk without winutils.
    # We will use toPandas() as a reliable fallback for small datasets to ensure the pipeline works.
    try:
        pdf = df.toPandas()
        pdf.to_csv(OUTPUT_FILE, index=False)
        print(f"SUCCESS: Feature engineering complete. Saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"FAILED: Could not convert to Pandas or save: {e}")

if __name__ == "__main__":
    print("Spark Processor Started")
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Input file {INPUT_FILE} not found.")
    else:
        try:
            process_data_spark()
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
        finally:
            if 'spark' in globals():
                spark.stop()
            print("Spark Processor Finished")
