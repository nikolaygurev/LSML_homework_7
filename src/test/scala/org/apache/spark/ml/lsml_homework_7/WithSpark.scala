package org.apache.spark.ml.lsml_homework_7

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.lsml_homework_7.WithSpark._sqlc
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StructType}


trait WithSpark {
  lazy val spark: SparkSession = WithSpark._spark
  lazy val sqlc: SQLContext = WithSpark._sqlc

  lazy val schema: StructType = new StructType()
    .add("feature_1", DoubleType)
    .add("feature_2", DoubleType)
    .add("feature_3", DoubleType)
    .add("feature_4", DoubleType)
    .add("target", DoubleType)

  lazy val fileName = "src/test/scala/org/apache/spark/ml/lsml_homework_7/lsml_homework_7.csv"
  lazy val path: String = getClass.getResource(fileName).getPath
  lazy val dfOrigin: DataFrame = _sqlc.read.option("header", "true").schema(schema).csv(path)

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