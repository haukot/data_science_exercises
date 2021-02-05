package experiments

import another.Utils
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.rdd.RDD

/**
 * Created by haukot on 11.10.15.
 */
object DecisionTreeCompensitionCategory extends App  {
  val sc = Utils.getSparkContext("LogisticRegression")

  var columnsAction = List[(List[Integer], String)]()
  columnsAction = (Array(1,2,3,4,5,6,7,9,11,12,14,15,16,19,22,34).map ( x => new Integer(x) ).toList, "UniqIndexInColumn") :: columnsAction

  var data = Utils.getAndProcessData(sc, "data/train.csv", "data/test.csv", "LongFromHex", "EmptiesToBillion", columnsAction)
  var (allTrainData, resData) = Utils.getTrainAndTestLabeledPointData(data)

  var firstCategoryData = Utils.replaceLabelsOnTwoCategory(data, "1.0")
  var (firstCategoryAllTrain, resData2) = Utils.getTrainAndTestLabeledPointData(firstCategoryData)

  val splits = allTrainData.randomSplit(Array(0.7, 0.3), seed = 11L)
  val trainData = splits(0).cache()
  val testData = splits(1).cache()

  val firstCategorySplits = firstCategoryAllTrain.randomSplit(Array(0.6, 0.4), seed = 11L)
  val firstCategoryTrainData = firstCategorySplits(0).cache()
  val firstCategoryTestData = firstCategorySplits(1).cache()

  // Run training algorithm to build the model
  //val numIterations = 1000
  //val model = SVMWithSGD.train(trainData, numIterations)

  var categoricalFeaturesInfo: Map[Int, Int] = Map[Int, Int]()
  categoricalFeaturesInfo += 1 -> 18
  categoricalFeaturesInfo += 2 -> 3
  categoricalFeaturesInfo += 3 -> 3
  categoricalFeaturesInfo += 4 -> 6
  categoricalFeaturesInfo += 5 -> 7
  categoricalFeaturesInfo += 6 -> 2
  categoricalFeaturesInfo += 7 -> 2
  categoricalFeaturesInfo += 9 -> 9
  categoricalFeaturesInfo += 11 -> 5
  categoricalFeaturesInfo += 12 -> 8
  categoricalFeaturesInfo += 14 -> 44
  categoricalFeaturesInfo += 15 -> 3
  categoricalFeaturesInfo += 16 -> 12
  categoricalFeaturesInfo += 19 -> 15
  categoricalFeaturesInfo += 22 -> 10
  categoricalFeaturesInfo += 34 -> 3

  val numClasses = 7
  val impurity = "entropy"
  val maxDepth = 7
  val maxBins = 80

  var model = DecisionTree.trainClassifier(trainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);
  var modelFirstCategory = DecisionTree.trainClassifier(firstCategoryTrainData, 2, categoricalFeaturesInfo, "gini", 3, 105);
  Utils.printMetrics(modelFirstCategory, firstCategoryTestData, 2)

  println("Learned classification tree model:\n" + model.toDebugString)

  class MyModel extends Serializable
  val mymodel = new MyModel {
    def predict(features: org.apache.spark.mllib.linalg.Vector) = {
      if (modelFirstCategory.predict(features) == 0.0) {
        1.0
      } else {
        model.predict(features)
      }
    }
  }
  Utils.printMetrics(mymodel, testData, 7)
  model = DecisionTree.trainClassifier(allTrainData, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins);
  modelFirstCategory = DecisionTree.trainClassifier(firstCategoryAllTrain, 2, categoricalFeaturesInfo, "gini", 3, 105);
  val result = Utils.getLinearModelResult(mymodel, resData)
  Utils.forceSaveResult(result, "results/new_model")
}
