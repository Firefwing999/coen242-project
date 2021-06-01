import pyspark
from pyspark.sql import SparkSession
import sys
import argparse
import numpy as np
import pickle
from pprint import pprint
from pyspark.sql import SQLContext
from pyspark.ml.feature import OneHotEncoderEstimator

spark = SparkSession \
    .builder \
    .appName("Logistic Regression on Census Data") \
    .getOrCreate()


sc = spark.sparkContext
sqlContext = SQLContext(sc)


df=spark.read \
 .option("header","False")\
 .option("inferSchema","True")\
 .option("sep",",")\
 .csv("/user/jahn/coen242/adult.data")
print("There are",df.count(),"rows",len(df.columns),
      "columns" ,"in the data.")

#print(type(df))


# df.show(4)


filename = "headers.txt"
df.printSchema()


#Get the naming correct here for future reference"
# df.show(4)

encoder = OneHotEncoderEstimator()\
    .setInputCols(['INSERT COLUMNS HERE'])\
    .setOutputCols(['NAME OF NEW COLUMNS'])

encoder_model = encoder.fit()
encoder_df=encoder_model.transform()

encoder_df.toPandas().head()
