import another.Utils;

import java.util.*;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.rdd.RDD;
import scala.Tuple2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import scala.collection.JavaConversions;
import scala.collection.immutable.*;;


public class DecisionTreeClassification {
    public static void main(String[] args) {
        SparkContext sc = Utils.getSparkContext("DecisionTreeClassification", "local");

        // для преобразования колонок с потенциальными категориями в индексы
        List<Tuple2<scala.collection.immutable.List<Integer>, String>> columnsProcessList = new ArrayList<>();
        List<Integer> categoricalColumns = Arrays.asList(1,2,3,4,5,6,7,9,11,12,14,15,16,19,22,34);
        Tuple2<scala.collection.immutable.List<Integer>, String> categoric = new Tuple2<>(JavaConversions.asScalaBuffer(categoricalColumns).toList(), "UniqIndexInColumn");
        columnsProcessList.add(categoric);

        RDD<Tuple2<Object, scala.collection.immutable.List<String>>> partsData = Utils.getAndProcessData(sc, "data/train.csv", "data/test.csv", "LongFromHex", "EmptiesToBillion",
                JavaConversions.asScalaBuffer(columnsProcessList).toList());
        //partsData = Utils.replaceLabelsOnTwoCategory(partsData, "0.0");

        Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>> data = Utils.getTrainAndTestLabeledPointData(partsData);
        JavaRDD<LabeledPoint> allTrainData = data._1().toJavaRDD();
        JavaRDD<LabeledPoint> resData = data._2().toJavaRDD();

        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = allTrainData.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Set parameters.
        //  Empty categoricalFeaturesInfo indicates all features are continuous.
        Integer numClasses = 7;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<Integer, Integer>();
        categoricalFeaturesInfo.put(1, 18); //with empty
        categoricalFeaturesInfo.put(2, 3); //with empty
        categoricalFeaturesInfo.put(3, 3); //with empty
        categoricalFeaturesInfo.put(4, 6); //with empty
        categoricalFeaturesInfo.put(5, 7); //with empty
        categoricalFeaturesInfo.put(6, 2); // всё равно не учитываются в дереве
        categoricalFeaturesInfo.put(7, 2); // всё равно не учитываются в дереве
        categoricalFeaturesInfo.put(9, 9); //with empty
        categoricalFeaturesInfo.put(11, 5); //with empty
        categoricalFeaturesInfo.put(12, 8); //with empty
        categoricalFeaturesInfo.put(14, 44); //with empty
        categoricalFeaturesInfo.put(15, 3); //with empty
        categoricalFeaturesInfo.put(16, 12); //with empty
        categoricalFeaturesInfo.put(19, 15); //with empty
        categoricalFeaturesInfo.put(22, 10); //with empty - x
        categoricalFeaturesInfo.put(34, 3); //with empty - стопудово categorical
        String impurity = "entropy";

        Integer maxDepth = 7;
        Integer maxBins = 80;

        // Train a DecisionTreeClassification model for classification.
        final DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Double, Double> predictionAndLabel =
                testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(LabeledPoint p) {
                        return new Tuple2<Double, Double>(model.predict(p.features()), p.label());
                    }
                });
        Double testErr =
                1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
                    @Override
                    public Boolean call(Tuple2<Double, Double> pl) {
                        return !pl._1().equals(pl._2());
                    }
                }).count() / testData.count();

        Utils.printMetrics(model, testData.rdd(), 7);
        System.out.println("Test Error: " + testErr);
        System.out.println("Learned classification tree model:\n" + model.toDebugString());

        // получаем результат
        final DecisionTreeModel resModel = DecisionTree.trainClassifier(allTrainData, numClasses,
                categoricalFeaturesInfo, impurity, maxDepth, maxBins);
        RDD result = Utils.getLinearModelResult(resModel, resData.rdd());
        Utils.forceSaveResult(result, "results/decision_tree_result");
    }
}