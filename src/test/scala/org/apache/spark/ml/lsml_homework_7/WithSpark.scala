package org.apache.spark.ml.lsml_homework_7

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}


trait WithSpark {
  lazy val spark: SparkSession = WithSpark._spark
  lazy val sqlc: SQLContext = WithSpark._sqlc

  lazy val fileName = "src/test/scala/org/apache/spark/ml/lsml_homework_7/lsml_homework_7.csv"
  lazy val dfOrigin: DataFrame = sqlc.read.option("header", "true").option("inferSchema", "true").csv(fileName)


  lazy val df: DataFrame = new VectorAssembler()
    .setInputCols(Array("feature_1", "feature_2", "feature_3", "feature_4"))
    .setOutputCol("features")
    .transform(dfOrigin)
    .drop("feature_1", "feature_2", "feature_3", "feature_4")
}

object WithSpark {
  lazy val _spark: SparkSession = SparkSession.builder
    .appName("Simple Application")
    .master("local[4]")
    .getOrCreate()

  lazy val _sqlc: SQLContext = _spark.sqlContext
}