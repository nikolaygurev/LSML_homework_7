package org.apache.spark.ml.lsml_homework_7

import breeze.linalg.{sum, DenseVector => BreezeDenseVector}
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.regression.{RegressionModel, Regressor}
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, MetadataUtils}
import org.apache.spark.sql.{Dataset, Encoder, Row}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder


////////////////////////////////////////////////////////////////
// LinearRegressionParams
////////////////////////////////////////////////////////////////


trait LinearRegressionParams extends PredictorParams with HasFeaturesCol with HasLabelCol with HasPredictionCol {
  val numEpochs: IntParam = new IntParam(this, "numEpochs", "numEpochs")
  val learningRate: DoubleParam = new DoubleParam(this, "learningRate", "learningRate")

  setDefault(numEpochs, 100)
  setDefault(learningRate, 1.0)

  setDefault(featuresCol, "features")
  setDefault(labelCol, "label")
  setDefault(predictionCol, "prediction")

  def setnumEpochs(value: Int): this.type = set(numEpochs, value)

  def setLearningRate(value: Double): this.type = set(learningRate, value)
}


////////////////////////////////////////////////////////////////
// LinearRegression
////////////////////////////////////////////////////////////////


class LinearRegression(override val uid: String) extends Regressor[Vector, LinearRegression, LinearRegressionModel]
  with LinearRegressionParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override protected def train(dataset: Dataset[_]): LinearRegressionModel = {
    val numFeatures = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var weights: BreezeDenseVector[Double] = BreezeDenseVector.zeros(numFeatures + 1)

    val getGradientUdf = dataset.sqlContext.udf.register(uid + "_gradient",
      (features: Vector, yTrue: Double) => {
        val intercept = BreezeDenseVector(1.0)
        val fullFeatures = BreezeDenseVector.vertcat(features.asBreeze.toDenseVector, intercept)
        val yPred = sum(fullFeatures * weights)
        val gradient = fullFeatures * (2 * (yPred - yTrue))
        Vectors.fromBreeze(gradient)
      }
    )

    for (_ <- 0 to $(numEpochs)) {
      // set datasetFull = dataset
      val datasetFull = dataset.withColumn("gradient", getGradientUdf(dataset($(featuresCol)), dataset($(labelCol))))
      val Row(Row(gradient)) = datasetFull
        .select(Summarizer.metrics("mean").summary(datasetFull("gradient")))
        .first()

      val gradValue: BreezeDenseVector[Double] = gradient.asInstanceOf[DenseVector].asBreeze.toDenseVector
      weights = weights - $(learningRate) * gradValue
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights))).setParent(this)
  }


  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]


////////////////////////////////////////////////////////////////
// LinearRegressionModel
////////////////////////////////////////////////////////////////


class LinearRegressionModel private[lsml_homework_7](override val uid: String, val weights: Vector)
  extends RegressionModel[Vector, LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[lsml_homework_7] def this(weights: Vector) =
    this(Identifiable.randomUID("linearRegressionModel"), weights)


  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights))


  override def predict(features: Vector): Double = {
    val intercept = BreezeDenseVector(1.0)
    val fullFeatures = BreezeDenseVector.vertcat(features.asBreeze.toDenseVector, intercept)
    val prediction = sum(fullFeatures * weights.asBreeze.toDenseVector)
    prediction
  }


  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = Tuple1(weights.asInstanceOf[Vector])

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}


object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      // Used to convert untyped dataframes to datasets with vectors
      implicit val encoder: Encoder[Vector] = ExpressionEncoder()

      val (weights) = vectors.select(vectors("_1").as[Vector]).first()

      val model = new LinearRegressionModel(weights)
      metadata.getAndSetParams(model)
      model
    }
  }
}