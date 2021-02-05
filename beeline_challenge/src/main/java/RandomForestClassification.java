/**
 * Created by haukot on 11.10.15.
 */
import another.Utils;
import org.apache.spark.SparkContext;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import scala.collection.JavaConversions;
import scala.collection.immutable.List;

public class RandomForestClassification {
    public static void main(String[] args) {
        SparkContext sc = Utils.getSparkContext("DecisionTreeClassification", "local");

        // для преобразования колонок с потенциальными категориями в индексы
        java.util.List<Tuple2<List<Integer>, String>> columnsProcessList = new ArrayList<>();
        java.util.List<Integer> categoricalColumns = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 9, 11, 12, 14, 15, 16, 19, 22, 34);
        Tuple2<scala.collection.immutable.List<Integer>, String> categoric = new Tuple2<>(JavaConversions.asScalaBuffer(categoricalColumns).toList(), "UniqIndexInColumn");
        columnsProcessList.add(categoric);

        RDD<Tuple2<Object, scala.collection.immutable.List<String>>> partsData = Utils.getAndProcessData(sc, "data/train.csv", "data/test.csv", "LongFromHex", "EmptiesToBillion",
                JavaConversions.asScalaBuffer(columnsProcessList).toList());

        Tuple2<RDD<LabeledPoint>, RDD<LabeledPoint>> data = Utils.getTrainAndTestLabeledPointData(partsData);
        JavaRDD<LabeledPoint> allTrainData = data._1().toJavaRDD();
        JavaRDD<LabeledPoint> resData = data._2().toJavaRDD();

        // Split the data into training and test sets (30% held out for testing)
        JavaRDD<LabeledPoint>[] splits = allTrainData.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        // Train a RandomForest model.
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
        Integer maxBins = 70;
        Integer numTrees = 1; // Use more in practice.
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        Integer seed = 12345;

        final RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses,
                categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
                seed);

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
        System.out.println("Learned classification forest model:\n" + model.toDebugString());

        // получаем результат
        final RandomForestModel resModel = RandomForest.trainClassifier(allTrainData, numClasses,
                categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
                seed);
        System.out.println("Learned classification forest result model:\n" + resModel.toDebugString());
        RDD result = Utils.getLinearModelResult(resModel, resData.rdd());
        Utils.forceSaveResult(result, "results/decision_tree_result");
    }
}
