package another

import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, MulticlassMetrics}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Try
import scalax.file.Path

/**
 * Created by haukot on 07.10.15.
 */
object Utils {
  def parseDoubleToZero(s: String) = try { s.toDouble } catch { case _:Throwable => 0.0 }

  def isNotEmpty(x: String) = x != null && x.trim.nonEmpty

  def hex2dec(hex: String): BigInt = {
    hex.toLowerCase().toList.map(
      "0123456789abcdef".indexOf(_)).map(
      BigInt(_)).reduceLeft( _ * 16 + _)
  }

  def getSparkContext(appName: String = "", master: String = "local") = {
    val conf = new SparkConf().setAppName(appName).setMaster(master)
    val sc = new SparkContext(conf)

    sc
  }


  // пустые значения считаются за отдельную строку в этих колонках
  def replaceStringsToFrequencyInColumn(partsData: RDD[Tuple2[Double, List[String]]],
                                        columnIndexes: List[Int] = List.concat(0 to 5, 9 to 12, 14 to 22)) = {
    // подготавляваемся к приводу в double к нему строки(по количеству вхождений строки в столбце)
    // hash - номер колонки_строка => количество вхождений ( типо "0 fa32s123" => 23041 }
    val stringHash = new mutable.HashMap[String,Int]() { override def default(key:String) = 0 }
    partsData.collect.map { case (label, parts) =>
      // строковые колонки
      columnIndexes.map { x =>
        val str = parts(x)
        val key = f"$x $str"
        stringHash.put(key, stringHash(key) + 1) // считает количество вхождений
        key
      }
    } // с collect - хак, чтобы map выполнился сейчас(а то он ленивый походу). над глянуть, как лучше(если это возможно)

    partsData.map { case (label, parts) =>
      val newParts = List.concat(0 to 61).map { x =>
        if (columnIndexes.contains(x)) {
          // x - строка в этих колонках
          val str = parts(x)
          val key = f"$x $str"
          stringHash(key).toDouble.toString
        } else {
          parts(x)
        }
      }
      (label, newParts)
    }
  }


  // пустые значения считаются за отдельную строку в этих колонках
  def replaceStringsToUniqIndexInColumn(partsData: RDD[Tuple2[Double, List[String]]],
                                        columnIndexes: List[Int] = List.concat(0 to 5, 9 to 12, 14 to 22)) = {
    val stringHash1 = new mutable.HashMap[Int, mutable.ArrayBuffer[String]]()
    partsData.collect().map { case (label, parts) =>
      // строковые колонки
      columnIndexes.map { x =>
        if (stringHash1.contains(x)) {
          var list = stringHash1.get(x).get
          list += parts(x)
        } else {
          var list : mutable.ArrayBuffer[String] = mutable.ArrayBuffer(parts(x))
          stringHash1.put(x, list)
        }
      }
    } // с collect - хак, чтобы map выполнился сейчас(а то он ленивый походу). над глянуть, как лучше(если это возможно)

    val stringHash = new mutable.HashMap[Int, List[String]]()
    columnIndexes.map { x =>
      var list = stringHash1(x).toList.distinct
      stringHash.put(x, list)
    }

    partsData.map { case (label, parts) =>
      val newParts = List.concat(0 to 61).map { x =>
        if (columnIndexes.contains(x)) {
          // x - строка в этих колонках
          stringHash(x).indexOf(parts(x)).toString
        } else {
          parts(x)
        }
      }
      (label, newParts)
    }
  }


  def replaceStringsToLongFromHex(partsData: RDD[Tuple2[Double, List[String]]],
                                  columnIndexes: List[Int] = List.concat(0 to 5, 9 to 12, 14 to 22)) = {
    partsData.map { case (label, parts) =>
      val newParts = List.concat(0 to 61).map { x =>
        if (isNotEmpty(parts(x)) && columnIndexes.contains(x)) {
          // x - строка в этих колонках
          hex2dec(parts(x)).toString
        } else {
          parts(x)
        }
      }
      (label, newParts)
    }
  }


  def replaceLabelsOnTwoCategory(partsData: RDD[Tuple2[Double, List[String]]],
                                 oneCategory: String) = {
    val category = oneCategory.toDouble
    partsData.map { case (label, parts) =>
      if (label == -1.0) {
        (label, parts)
      } else if (label == category) {
        // -1.0 - test.csv label dummy
        (0.0, parts)
      } else {
        (1.0, parts)
      }
    }
  }


  def replaceEmptiesToZero(partsData: RDD[Tuple2[Double, List[String]]]) = {
    partsData.map { case (label, parts) =>
      val newParts = parts.map {
        case y if !isNotEmpty(y) => 0.0.toString
        case y => y
      }
      (label, newParts)
    }
  }


  def replaceEmptiesToBillion(partsData: RDD[Tuple2[Double, List[String]]]) = {
    partsData.map { case (label, parts) =>
      val newParts = parts.map {
        case y if !isNotEmpty(y) => 1000000000.0.toString
        case y => y
      }
      (label, newParts)
    }
  }


  def getLabeledData(sc: SparkContext, trainDataPath: String, resDataPath: String) = {
    val trainData = sc.textFile(trainDataPath).map { line =>
      val partsLine = line.split(",", -1)
      val label = partsLine(62).toDouble // 62 - столбец с Y
      (label, partsLine.toList)
    }
    val resData = sc.textFile(resDataPath).map { line =>
      var partsLine = line.split(",", -1)
      partsLine = partsLine.slice(1, partsLine.length) // отрезаем столбец с id
      // -1.0 - специально, чтобы различать train и test
      (-1.0, partsLine.toList)
    }

    var partsData = trainData.union(resData)

    partsData
  }


  def getTrainAndTestLabeledPointData(data: RDD[Tuple2[Double, List[String]]]) = {
    // подготавливаем данные в виде массива из {результирующий y, [остальные переменные]}
    val parsedData = data.map { case (label, parts) =>
      val vectors = (0 to 61).toList.map { x =>
        parts(x).toDouble
      }.toArray

      LabeledPoint(label, Vectors.dense(vectors))
    }

    val resultTrainData = parsedData.filter { x =>
      x.label != -1.0 // -1.0 - специально, чтобы различать train и test
    }
    val resultResData = parsedData.filter { x =>
      x.label == -1.0
    }
    (resultTrainData, resultResData)
  }


  // может применятся не только к строкам вобщем то, а к колонкам в целом
  def replaceColumns(partsData: RDD[Tuple2[Double, List[String]]], method: String,
                     columns: List[Int] = List.concat(0 to 5, 9 to 12, 14 to 22)) = {
    method match {
      case "FrequencyInColumn" => replaceStringsToFrequencyInColumn(partsData, columns)
      case "UniqIndexInColumn" => replaceStringsToUniqIndexInColumn(partsData, columns)
      case "LongFromHex" => replaceStringsToLongFromHex(partsData, columns)
      case null => partsData
    }
  }


  def replaceEmpties(partsData: RDD[Tuple2[Double, List[String]]], method: String) = {
    method match {
      case "EmptiesToZero" => replaceEmptiesToZero(partsData)
      case "EmptiesToBillion" => replaceEmptiesToBillion(partsData)
      case null => partsData
    }
  }


  // основная функции загрузки данных
  // dataType = ["data_with_y", "data_with_id"],
  def getAndProcessData(sc: SparkContext, trainDataPath: String, resDataPath: String, stringsReplaceType: String = null,
                         emptiesReplaceType: String = null, replaceColumnsData: List[(List[Integer], String)] = null) = {
    var partsData = getLabeledData(sc, trainDataPath, resDataPath)
    // применяем метод по замене строк
    partsData = replaceColumns(partsData, stringsReplaceType)

    // заменяем определенные колонки
    if (replaceColumnsData != null) {
      replaceColumnsData.foreach { case (columns, method) =>
        var intColumns = columns.map(x => x.toInt)
        partsData = replaceColumns(partsData, method, intColumns)
      }
    }
    // применяем метод по замене пропусков
    partsData = replaceEmpties(partsData, emptiesReplaceType)
    partsData
  }


  def getLinearModelResult[Model <: { def predict(features: org.apache.spark.mllib.linalg.Vector): Double }](model: Model, resData: RDD[LabeledPoint]) = {
    resData.zipWithIndex.map { case (LabeledPoint(label, features), i) =>
      var prediction = model.predict(features).toInt
      f"$i,$prediction"
    }
  }

  def forceSaveResult(result: RDD[String], savePath: String) = {
    // remove old directory
    val path: Path = Path.fromString(savePath)
    Try(path.deleteRecursively(continueOnFailure = false))

    saveResult(result, savePath)
  }

  def saveResult(result: RDD[String], savePath: String) = {
    result.saveAsTextFile(savePath)
  }

  def getMetrics[Model <: { def predict(features: org.apache.spark.mllib.linalg.Vector): Double }](model: Model, data: RDD[LabeledPoint]):
    MulticlassMetrics = {
    val predictionsAndLabels = data.map(example =>
      (model.predict(example.features), example.label)
    )
    new MulticlassMetrics(predictionsAndLabels)
  }

  def printMetrics[Model <: { def predict(features: org.apache.spark.mllib.linalg.Vector): Double }](model: Model, data: RDD[LabeledPoint], categoriesLength: Int = 7) = {
    val metrics = getMetrics(model, data)
    (0 until categoriesLength).map(
        cat => (cat, metrics.precision(cat), metrics.recall(cat))
    ).foreach(println)
    println("Precision of categories ^ ")
  }

  def testMSE(valuesAndPreds: RDD[Tuple2[Double, Double]]) = {
    val MSE = valuesAndPreds.map{case(v, p) => math.pow((v - p), 2)}.mean()

    MSE
  }

  def testMulticlassPrecison(predictionAndLabels: RDD[Tuple2[Double, Double]]) = {
    // Get evaluation metrics.
    val metrics = new MulticlassMetrics(predictionAndLabels)
    val precision = metrics.precision

    precision
  }
}
