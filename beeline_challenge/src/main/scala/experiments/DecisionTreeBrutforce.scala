package experiments


import another.Utils
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils

import scala.collection.mutable

/**
 * Created by haukot on 11.10.15.
 */
object DecisionTreeBrutforce extends App {
  val sc = Utils.getSparkContext("LogisticRegression")

  var columnsAction = List[(List[Integer], String)]()
  columnsAction = (Array(1,2,3,4,5,6,7,9,11,12,14,15,16,19,22,34).map ( x => new Integer(x) ).toList, "UniqIndexInColumn") :: columnsAction
  var data = Utils.getAndProcessData(sc, "data/train.csv", "data/test.csv", "LongFromHex", "EmptiesToBillion", columnsAction)
  data = Utils.replaceLabelsOnTwoCategory(data, "1.0") // gini 3 105 for 1.0 - 0.793, 0.9659
  var (allTrainData, resData) = Utils.getTrainAndTestLabeledPointData(data)

  val splits = allTrainData.randomSplit(Array(0.6, 0.4), seed = 11L)
  val trainData = splits(0).cache()
  val cvData = splits(1).cache()

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

  val evaluations =
    for (impurity <- Array("gini", "entropy");
         depth    <- Array(2,3,4, 5, 6, 7);
         bins     <- (75 to 120 by 10).toArray)
//         depth    <- (6 to 12).toArray;
//         bins     <- (40 to 120 by 10).toArray)
      yield {
        val model = DecisionTree.trainClassifier(
          trainData, 2, categoricalFeaturesInfo, impurity, depth, bins)
        val predictionsAndLabels = cvData.map(example =>
          (model.predict(example.features), example.label)
        )
        val metrics = new MulticlassMetrics(predictionsAndLabels)
        val accuracy = metrics.precision(0.0)
        ((impurity, depth, bins), accuracy, metrics.precision(1.0))
      }

  evaluations.sortBy(_._2).reverse.foreach(println)
//   Итоговый топ:
//  ((gini,8,80),0.7444240226516318)
//  ((entropy,7,80),0.7440763002334707)
//  ((entropy,8,80),0.7432815061348169)
//  ((entropy,7,70),0.7429337837166559)
//  ((entropy,7,65),0.7427847598231583)
//  ((entropy,8,55),0.7426854105608266)
//  ((gini,7,80),0.7422880135114996)
//  ((gini,7,60),0.7419899657245045)
//  ((entropy,7,75),0.7418409418310069)
//  ((gini,7,55),0.7417415925686752)
//  ((entropy,8,65),0.7416919179375093)
}
