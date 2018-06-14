import org.apache.spark.rdd._
import scala.collection.JavaConverters._
import au.com.bytecode.opencsv.CSVReader
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import java.io._
import org.joda.time._
import org.joda.time.format._


import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.regression.LinearRegressionModel

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object log_reg_origin_dest extends App {
 val conf = new SparkConf().setAppName("Airline Delay Prediction by Arrival Time").setMaster("local[2]").set("spark.executor.memory","1g");
val sc = new SparkContext(conf)
case class FlightDelay(year: String,
                      month: String,
                      dayOfMonth: String,
                      dayOfWeek: String,
                      crsDepTime: String,
                      depDelay: String,
                      origin: String,
                     destination: String,
                      distance: String,
                      cancelled: String) {
    
def feature_details: (String, Array[Double]) = {
      val values = Array(
        depDelay.toDouble,
        month.toDouble,
        dayOfMonth.toDouble,
        dayOfWeek.toDouble,
        get_hour(crsDepTime).toDouble,
        distance.toDouble
       )
       
     new Tuple2(get_date(year.toInt, month.toInt, dayOfMonth.toInt), values)
    }

    def get_hour(depTime: String) : String = "%04d".format(depTime.toInt).take(2)
    def get_date(year: Int, month: Int, day: Int) = "%04d%02d%02d".format(year, month, day)

  }

// function to do a preprocessing step for a given file
def prepFlightDelays(infile: String): RDD[FlightDelay] = {
    val data = sc.textFile(infile)

    data.map { line =>
      val reader = new CSVReader(new StringReader(line))
      reader.readAll().asScala.toList.map(rec => FlightDelay(rec(1),rec(2),rec(3),rec(4),rec(7),rec(8),rec(5),rec(6),rec(12),rec(11)))
    }.map(list => list(0))
    .filter(rec => rec.year != "Year")
    .filter(rec => rec.cancelled == "0")
  .filter(rec => rec.origin == "DFW")
  .filter(rec => rec.destination == "JFK")
}

val data_1415 = prepFlightDelays("airline/Dataset2015_1.csv").map(rec => rec.feature_details._2)
val data_2016 = prepFlightDelays("airline/Dataset2016_1.csv").map(rec => rec.feature_details._2)
data_1415.take(5).map(x => x mkString ",").foreach(println)

def parseData(vals: Array[Double]): LabeledPoint = {
  LabeledPoint(if (vals(0)>=15) 1.0 else 0.0, Vectors.dense(vals.drop(1)))
}

// Prepare training set
val parsedTrainingData = data_1415.map(parseData)
parsedTrainingData.cache
val scaler = new StandardScaler(withMean = true, withStd = true).fit(parsedTrainingData.map(x => x.features))
val scaledTrainingData = parsedTrainingData.map(x => LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray))))
scaledTrainingData.cache

// Prepare test/validation set
val parsedTestingData = data_2016.map(parseData)
parsedTestingData.cache
val scaledTestingData = parsedTestingData.map(x => LabeledPoint(x.label, scaler.transform(Vectors.dense(x.features.toArray))))
scaledTestingData.cache

scaledTestingData.take(3).map(x => (x.label, x.features)).foreach(println)

scaledTrainingData.take(3).map(x => (x.label, x.features)).foreach(println)
//scaledTrainingData.saveAsTextFile("Scaled_Traindata")
def metricsToEvaluate(labelsPrediction: RDD[(Double, Double)]) : Tuple2[Array[Double], Array[Double]] = {
    val tp = labelsPrediction.filter(r => r._1==1 && r._2==1).count.toDouble
    val tn = labelsPrediction.filter(r => r._1==0 && r._2==0).count.toDouble
    val fp = labelsPrediction.filter(r => r._1==1 && r._2==0).count.toDouble
    val fn = labelsPrediction.filter(r => r._1==0 && r._2==1).count.toDouble

    val precision = tp / (tp+fp)
    val recall = tp / (tp+fn)
    val F_measure = 2*precision*recall / (precision+recall)
    val accuracy = (tp+tn) / (tp+tn+fp+fn)
    new Tuple2(Array(tp, tn, fp, fn), Array(precision, recall, F_measure, accuracy))
}
val model_lr = LogisticRegressionWithSGD.train(scaledTrainingData, numIterations=100)

// Predict
val valuesAndPreds_lr = scaledTestingData.map { point =>
    val pred = model_lr.predict(point.features)
   // (pred, point.label)
    (point.label, point.features, pred)
}
val valuesAndPreds_lr1 = scaledTestingData.map { point =>
    val pred = model_lr.predict(point.features)
   (pred, point.label)
  // (point.label, point.features, pred)
}

val metrics = new MulticlassMetrics(valuesAndPreds_lr1)
val metrics1 = new BinaryClassificationMetrics(valuesAndPreds_lr1)



//Print Predicted and actual
valuesAndPreds_lr.take(10).foreach({case (v, f, p) => 
     println(s"Features: ${f}, Predicted: ${p}, Actual: ${v}")})

//For Accuracy
val m_lr = metricsToEvaluate(valuesAndPreds_lr1)._2
println("precision = %.2f, recall = %.2f, F1 = %.2f, accuracy = %.2f".format(m_lr(0), m_lr(1), m_lr(2), m_lr(3)))

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)


val precision = metrics1.precisionByThreshold
precision.foreach { case (t, p) =>
  println(s"Threshold: $t, Precision: $p")
}
val auPRC = metrics1.areaUnderPR
println("Area under precision-recall curve = " + auPRC)

// Compute thresholds used in ROC and PR curves
val thresholds = precision.map(_._1)

// ROC Curve
val roc = metrics1.roc

// AUROC
val auROC = metrics1.areaUnderROC
println("Area under ROC = " + auROC)
}