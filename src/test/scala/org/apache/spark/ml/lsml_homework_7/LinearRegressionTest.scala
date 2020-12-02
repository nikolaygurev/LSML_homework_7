package org.apache.spark.ml.lsml_homework_7

import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  val delta = 0.0001


  private def validateModel(model: LinearRegressionModel, data: DataFrame): Unit = {
    val dfTransformed = model.transform(df)

    val evaluator = new RegressionEvaluator().setLabelCol("target").setPredictionCol("prediction").setMetricName("mse")
    val mse = evaluator.evaluate(dfTransformed)

    mse should be < delta
  }


  "Model" should "make accurate predictions" in {
    val model = new LinearRegression().fit(df)
    validateModel(model, df)
  }


  "Estimator" should "learn weights correctly" in {
    val model = new LinearRegression().fit(df)
    val weights = model.weights.asBreeze.toDenseVector

    weights(0) should be(1.0 +- delta) // feature_1
    weights(1) should be(2.0 +- delta) // feature_2
    weights(2) should be(-1.0 +- delta) // feature_3
    weights(3) should be(-1.0 +- delta) // feature_4
    weights(4) should be(0.0 +- delta) // intercept
  }


  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(new LinearRegression()))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(df).stages(0).asInstanceOf[LinearRegressionModel]
    validateModel(model, df)
  }


  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(new LinearRegression()))
    val model = pipeline.fit(df)

    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)
    validateModel(reRead.stages(0).asInstanceOf[LinearRegressionModel], df)
  }
}