package experiments

import another.Utils
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by haukot on 07.10.15.
 */

// result = 52.75%
object LogisticMulticlassRegression extends App {
  // создаем контекст спарка (sc)
  val sc = Utils.getSparkContext("LogisticRegression")

  var data = Utils.getAndProcessData(sc, "data/train.csv", "data/test.csv", "UniqIndexInColumn", "EmptiesToBillion")
  var (allTrainData, resData) = Utils.getTrainAndTestLabeledPointData(data)

  val splits = allTrainData.randomSplit(Array(0.6, 0.4), seed = 11L)
  val training = splits(0).cache()
  val test = splits(1)

  // Building the model
  //val numIterations = 1 // 10
  val model = new LogisticRegressionWithLBFGS()
    .setNumClasses(7)

  val trainingModel = model.run(training)
  // Compute raw scores on the test set.
  val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
    val prediction = trainingModel.predict(features)
    (prediction, label)
  }

  val precision = Utils.testMulticlassPrecison(predictionAndLabels)
  println("Precision = " + precision)

  val resModel = model.run(allTrainData)
  val result = Utils.getLinearModelResult(resModel, resData)
  Utils.forceSaveResult(result, "results/logistic_regression_result")

  // Save and load model
  //model.save(sc, "myModelPath")
  //val sameModel = LogisticRegressionModel.load(sc, "myModelPath")
}

