package experiments

import another.Utils
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.linalg.Vectors

import scala.collection.mutable

/**
 * Created by haukot on 07.10.15.
 */

object LinearRegression extends App {
  // создаем контекст спарка (sc)
  val sc = Utils.getSparkContext("LinearRegression")

  var data = Utils.getAndProcessData(sc, "data/train.csv", "data/test.csv", "UniqIndexInColumn", "EmptiesToBillion")
  var (trainData, resData) = Utils.getTrainAndTestLabeledPointData(data)

  // Building the model
  val numIterations = 1 // 10
  val model = LinearRegressionWithSGD.train(trainData, numIterations)

  // Evaluate model on training examples and compute training error
  val valuesAndPreds = trainData.map { point =>
    val prediction = model.predict(point.features)
    (point.label, prediction)
  }

  val MSE = Utils.testMSE(valuesAndPreds)
  println("training Mean Squared Error = " + MSE)

  // Save and load model
  //model.save(sc, "myModelPath")
  //val sameModel = LinearRegressionModel.load(sc, "myModelPath")
}
