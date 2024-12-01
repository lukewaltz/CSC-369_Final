import org.apache.spark.SparkContext._
import scala.io._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection._
import java.io._

object Main {

  // Create empty files to record model performance
  logToFile("k_accuracy.csv", "k,Accuracy", false)
  logToFile("predictions.csv", "ID,Actual,Prediction", false)

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    System.setProperty("hadoop.home.dir", "C:/winutils/")

    val data = loadAndNormalizeData()

    // Split the dataset
    val Array(trainingData, validationData, testData) = data.randomSplit(Array(0.6, 0.2, 0.2), seed = 42)

    // Tune k using the validation set
    val kStart = 3      // Start value of k
    val kEnd = 30       // End value of k
    val kStep = 2       // Step size for k values
    val kValues = kStart to kEnd by kStep

    val bestK = tuneK(trainingData, validationData, kValues)
    println(s"Best k value: $bestK")

    // Evaluate the model on the test set
    val testAccuracy = evaluateModel(trainingData, testData, bestK, true)
    println(f"Test Accuracy: $testAccuracy%.2f%%")
  }

  private def loadAndNormalizeData(): RDD[(String, Int, Array[Double])] = {
    val conf = new SparkConf()
      .setAppName("Cancer KNN")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    val datasetPath = "Cancer_Data.csv"
    val rawRDD = sc.textFile(datasetPath)

    // Split header and data
    val header = rawRDD.first()
    val data = rawRDD.filter(row => row != header)

    // Split rows into columns
    val splitData = data.map(row => row.split(","))

    // Parse data into (id, diagnosis, features)
    val parsedData = splitData.map(row => {
      val id = row(0)
      val diagnosis = if (row(1) == "M") 1 else 0 // Convert diagnosis to 1 (Malignant) and 0 (Benign)
      val features = row.slice(2, row.length).map(_.toDouble) // Convert features to Double
      (id, diagnosis, features)
    })

    // Calculate Min and Max for each feature
    val features = parsedData.map(_._3) // Extract feature arrays
    val featureMin = features.reduce((a, b) => a.zip(b).map { case (x, y) => math.min(x, y) })
    val featureMax = features.reduce((a, b) => a.zip(b).map { case (x, y) => math.max(x, y) })

    // Normalize features using Min-Max scaling
    parsedData.map { case (id, diagnosis, features) =>
      val normalizedFeatures = features.zip(featureMin.zip(featureMax)).map {
        case (value, (min, max)) =>
          if (max != min) (value - min) / (max - min) else 0.0
      }
      (id, diagnosis, normalizedFeatures)
    }
  }

  private def classifyKNN(trainingData: RDD[(String, Int, Array[Double])], testData: RDD[(String, Int, Array[Double])], k: Int): RDD[(String, Int)] = {
    // Pair each test point with all training points (Cartesian product)
    val cartesianData = testData.cartesian(trainingData)

    // Compute distances for all pairs
    val distances = cartesianData.map { case ((testId, _, testFeatures), (_, trainingLabel, trainingFeatures)) =>
      val distance = math.sqrt(testFeatures.zip(trainingFeatures).map {
        case (a, b) => math.pow(a - b, 2)
      }.sum)
      (testId, (distance, trainingLabel))
    }

    // Group by test point and find k-nearest neighbors
    val kNearestNeighbors = distances.groupByKey().mapValues { neighbors =>
      neighbors.toSeq.sortBy(_._1).take(k) // Sort by distance and take the k nearest
    }

    // Predict the majority class for each test point
    kNearestNeighbors.mapValues { neighbors =>
      neighbors.groupBy(_._2) // Group by label
        .mapValues(_.size) // Count occurrences of each label
        .maxBy(_._2) // Select the label with the maximum count
        ._1
    }
  }

  private def tuneK(trainingData: RDD[(String, Int, Array[Double])], validationData: RDD[(String, Int, Array[Double])], kValues: Seq[Int]): Int = {
    val accuracyPerK = kValues.map { k =>
      val accuracy = evaluateModel(trainingData, validationData, k, false)
      logToFile("k_accuracy.csv", s"$k,$accuracy", true)
      (k, accuracy)
    }
    val bestK = accuracyPerK.maxBy(_._2)._1
    bestK
  }

  private def evaluateModel(trainingData: RDD[(String, Int, Array[Double])], testData: RDD[(String, Int, Array[Double])], k: Int, mode: Boolean): Double = {
    // Predict labels for the test data
    val predictions = classifyKNN(trainingData, testData, k)

    // Join predictions with actual labels
    val joinedData = testData.map { case (id, actual, _) => (id, actual) }.join(predictions)

    if (mode) {
      joinedData.collect().foreach({
        case (id, (actual, predicted)) => logToFile("predictions.csv", s"$id,$actual,$predicted", true)
      })
    }

    // Compute accuracy
    val correctPredictions = joinedData.filter { case (_, (actual, predicted)) => actual == predicted }.count()
    val totalPredictions = testData.count()

    val accuracy = (correctPredictions.toDouble / totalPredictions) * 100
    accuracy
  }

  def logToFile(filePath: String, data: String, mode: Boolean): Unit = {
    val writer = new FileWriter(filePath, mode) // Append mode
    try {
//      println(data)
      writer.write(data + "\n")
    } finally {
      writer.close()
    }
  }
}
