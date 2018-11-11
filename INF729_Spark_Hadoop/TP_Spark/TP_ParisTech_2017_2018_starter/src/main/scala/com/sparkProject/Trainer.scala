package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.CountVectorizer
import org.apache.spark.ml.feature.IDF
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoder
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")

    var df = spark.read.parquet("/home/jacky/Jacky_Kaub/INF729_Spark_Hadoop/TP_Spark/prepared_trainingset/")

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    val remover = new StopWordsRemover()
        .setInputCol("tokens")
        .setOutputCol("filtered")

    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vect")

    val idf = new IDF()
      .setInputCol("vect")
      .setOutputCol("tfidf")

    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")


    var encoderCountry = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_onehot")

    var encoderCurrency = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_onehot")


    val assembler = new VectorAssembler()
      .setInputCols(
        Array(
          "tfidf",
          "days_campaign",
          "hours_prepa",
          "goal",
          "country_onehot",
          "currency_onehot"
        )
      )
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          tokenizer,
          remover,
          cvModel,
          idf,
          indexerCountry,
          indexerCurrency,
          encoderCountry,
          encoderCurrency,
          assembler,
          lr
        )
      )


    val model =  pipeline.fit(df)


    println(s"Total number of rows: ${df.count}")
    println(s"Number of columns ${df.columns.length}")

    df.show()

  }
}
