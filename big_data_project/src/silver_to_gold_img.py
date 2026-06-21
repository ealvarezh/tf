import json
import os
import sys
import urllib.request
import io
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType, BooleanType, ArrayType, MapType, LongType
import os 
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ["HADOOP_HOME"] = rf"C:\Users\jc.ruedah\hadoop"
os.environ["PATH"] += r";C:\Users\jc.ruedah\hadoop\bin"
#Spark session
spark = (
    SparkSession.builder
    .appName("IMG_SILVER_TO_GOLD")
    .master("local[*]")
    # .config("spark.hadoop.hadoop.native.lib","false")
    # .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.RawLocalFileSystem")
    # .config("spark.hadoop.fs.file.impl.disable.cache","true")
    .getOrCreate()
)

def main():


    ruta_silver_distribuido = Path(__file__).parent.parent / "data" / "silver" / "embeddings_imagenes_distribuidas" / "dino"
    ruta_gold_distribuido = Path(__file__).parent.parent / "data" / "gold" / "embeddings_imagenes_distribuidas" / "dino"
    os.makedirs(ruta_gold_distribuido, exist_ok=True)
    df = spark.read.parquet(str(ruta_silver_distribuido))
    df.repartition(100).write.mode("overwrite").parquet(str(ruta_gold_distribuido))
    
    print("Proceso de migración de silver a gold completado exitosamente.")

if __name__ == "__main__":    main()
