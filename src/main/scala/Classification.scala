import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.functions.hour
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics

object Classification extends App {


  import org.apache.log4j._
  Logger.getLogger("org").setLevel(Level.ERROR)


  // Create Session App
  val spark = SparkSession.builder().master("local")
    .appName("ClassificationExample")
    .getOrCreate()

  // Use Spark to read in the Ecommerce_customers csv file.
  val data = spark.read.option("header","true").option("inferSchema","true")
    .format("com.databricks.spark.csv").load("src/main/resources/advertising.csv")


  // Print the Schema of the DataFrame
  data.printSchema()


  ////////////////////////////////////////////////////
  //// Setting Up DataFrame for Machine Learning ////
  //////////////////////////////////////////////////

  //   Do the Following:
  //    - Rename the Clicked on Ad column to "label"
  //    - Grab the following columns "Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Timestamp","Male"
  //    - Create a new column called Hour from the Timestamp containing the Hour of the click



  val dfTime=data.withColumn("Hour",hour(data("Timestamp")))

  // This is needed to use the $-notation
  import spark.implicits._
  val logRegData = dfTime.select(data("Clicked on Ad").as("label"), $"Daily Time Spent on Site",$"Age", $"Area Income", $"Daily Internet Usage",
                                      $"Hour", $"Male")


  val assembler = new VectorAssembler()
                  .setInputCols(Array("Daily Time Spent on Site", "Age", "Area Income","Daily Internet Usage", "Hour"))
                  .setOutputCol("features")


  // Use randomSplit to create a train test split of 70/30
  val Array(training, test) = logRegData.randomSplit(Array(0.7, 0.3), seed = 12345)

  ///////////////////////////////
  // Set Up the Pipeline ///////
  /////////////////////////////

  // Create a new LogisticRegression object called lr
  val lr = new LogisticRegression()

  // Create a new pipeline with the stages: assembler, lr
  val pipeline = new Pipeline().setStages(Array(assembler, lr))

  // Fit the pipeline to training set.
  val model = pipeline.fit(training)

  // Get Results on Test Set with transform
  val results = model.transform(test)

  ////////////////////////////////////
  //// MODEL EVALUATION /////////////
  //////////////////////////////////


  // Convert the test results to an RDD using .as and .rdd
  val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd

  // Instantiate a new MulticlassMetrics object
  val metrics = new MulticlassMetrics(predictionAndLabels)

  // Print out the Confusion matrix
  println("Confusion matrix:")
  println(metrics.confusionMatrix)


  // Stop spark
  spark.stop()
}
