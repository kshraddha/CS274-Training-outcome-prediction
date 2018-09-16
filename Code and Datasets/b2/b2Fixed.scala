import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import java.text.Normalizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import java.io._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructType,StructField,StringType,DoubleType};
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.VectorSlicer
import org.apache.spark.ml.feature.StringIndexer


val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._

def textNormalize(input: String): String = {
  
  // Text Normalization
  var inputNormalized = Normalizer.normalize(input, Normalizer.Form.NFD)
 
  // Remove Diacritical Marks
  inputNormalized = inputNormalized.replaceAll("[^\\p{ASCII}]", "")

  // Replace punctuation marks with space
  inputNormalized = inputNormalized.replaceAll("[^-a-zA-Z0-9]", " ")

  // Remove extra whitespaces
  inputNormalized = inputNormalized.replaceAll("\\s+", " ")

  // Covert all characters to lower case
  inputNormalized.toLowerCase()
 // Feature Transformation using spark

} 


val df = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("b2/datasets/b2dataset.csv")

var rowRDD = df.select( "dog_SubStatusCode","Age","Health","RespondsToCommandKennel","GoodWStrangers","TrafficFear","StaysOnCommand","BehavesWellClass","Sex","Breed", "DayInLife").rdd

val trainingRowRDD = sc.makeRDD(rowRDD.take((rowRDD.count * 0.8).toInt))
print("------------log----------------\n# of training data: " + trainingRowRDD.count + "\n------------log----------------\n")
val testingRowRDD = rowRDD.subtract(trainingRowRDD)
print("------------log----------------\n# of testing data: " + testingRowRDD.count + "\n------------log----------------\n")

val trainingData = trainingRowRDD.map{
case Row( label:String,num1:String,num2:String,num3:String,num4:String,num5:String,num6:String,num7:String, text1:String, text2:String, text3:String) => (label, num1, num2, num3, num4, num5, num6, num7, text1, text2, textNormalize(text3))
}.toDF("label","num1","num2","num3","num4","num5","num6","num7","text1","text2","text3"
).withColumn("label", 'label cast DoubleType
).withColumn("num1", 'num1 cast DoubleType
).withColumn("num2", 'num2 cast DoubleType
).withColumn("num3", 'num3 cast DoubleType
).withColumn("num4", 'num4 cast DoubleType
).withColumn("num5", 'num5 cast DoubleType
).withColumn("num6", 'num6 cast DoubleType
).withColumn("num7", 'num7 cast DoubleType)

val testData = testingRowRDD.map{
case Row( label:String,num1:String,num2:String,num3:String,num4:String,num5:String,num6:String,num7:String, text1:String, text2:String, text3:String) => (label, num1, num2, num3, num4, num5, num6, num7, text1, text2, textNormalize(text3))
}.toDF("label","num1","num2","num3","num4","num5","num6","num7","text1","text2","text3"
).withColumn("label", 'label cast DoubleType
).withColumn("num1", 'num1 cast DoubleType
).withColumn("num2", 'num2 cast DoubleType
).withColumn("num3", 'num3 cast DoubleType
).withColumn("num4", 'num4 cast DoubleType
).withColumn("num5", 'num5 cast DoubleType
).withColumn("num6", 'num6 cast DoubleType
).withColumn("num7", 'num7 cast DoubleType)


val indexer1 = new StringIndexer().setInputCol("text1").setOutputCol("text1Index")

val indexer2 = new StringIndexer().setInputCol("text2").setOutputCol("text2Index")

val tokenizer1 = new Tokenizer().setInputCol("text3").setOutputCol("raw3")

val remover1 = new StopWordsRemover().setInputCol("raw3").setOutputCol("filtered3")

val hashingTF1 = new HashingTF().setInputCol("filtered3").setOutputCol("rawfeatures3").setNumFeatures(2048)

val idf1 = new IDF().setInputCol("rawfeatures3").setOutputCol("features3")


val vectorA = new VectorAssembler().setInputCols(Array("num1","num2","num3","num4","num5","num6","num7", "text1Index", "text2Index", "features3")).setOutputCol("features")

val lr = new LogisticRegression().setMaxIter(20).setRegParam(1.1)

val pipeline = new Pipeline().setStages(Array (indexer1, indexer2, tokenizer1, remover1, hashingTF1, idf1, vectorA, lr))

val model = pipeline.fit(trainingData) 

val trainoutput = model.transform(trainingData)
val testoutput = model.transform(testData)

var trainingError = trainoutput.filter(r => r(0) == r(20)).count.toDouble / trainingData.count
var testingError = testoutput.filter(r => r(0) == r(20)).count.toDouble / testData.count

printf("Result using Fixed Split\n")
printf("Prediction reliability on trained data = %.2f%%\n", (100*trainingError))
printf("Prediction reliability on testing data = %.2f%%\n", (100*testingError))

