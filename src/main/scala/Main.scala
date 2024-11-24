import org.apache.spark.SparkContext._
import scala.io._
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection._

object Main {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    System.setProperty("hadoop.home.dir", "C:/winutils/")

    loadAndNormalizeData()
  }

  private def loadAndNormalizeData(): Unit = {

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

    // Normalize features using Min-Max scaling (lessens the impact of outliers on data)
    val normalizedData = parsedData.map { case (id, diagnosis, features) =>
      val normalizedFeatures = features.zip(featureMin.zip(featureMax)).map {
        case (value, (min, max)) =>
          if (max != min) (value - min) / (max - min) else 0.0
      }
      (id, diagnosis, normalizedFeatures)
    }

    //normalized features in order:
    normalizedData.take(5).foreach { case (id, diagnosis, features) =>
      println(s"ID: $id, Diagnosis: $diagnosis, Normalized Features: ${features.mkString(", ")}")
    }
  }
}