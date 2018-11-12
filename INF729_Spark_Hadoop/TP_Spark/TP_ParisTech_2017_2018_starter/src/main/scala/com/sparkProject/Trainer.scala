package com.sparkProject

//import des librairies
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
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}


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

    //Lecture du trainingset
    val df = spark.read.parquet("/home/jacky/Jacky_Kaub/INF729_Spark_Hadoop/TP_Spark/prepared_trainingset/")

    //On sépare le dataframe en train (90%) et test (10%)
    val splits = df.randomSplit(Array(0.9, 0.1), seed = 2904)

    //Training set :
    val training = splits(0).cache()

    //Test set :
    val test = splits(1)

    //On transforme le texte en liste
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //On retire les stopwords des listes créées précédement
    val remover = new StopWordsRemover()
        .setInputCol("tokens")
        .setOutputCol("filtered")

    //On vectorise la liste filtrée
    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vect")

    //On applique l'IDF à la colonne crée précédement
    val idf = new IDF()
      .setInputCol("vect")
      .setOutputCol("tfidf")

    //On catégorise en indexe la colonne country2
    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    //On catégorise en indexe la colonne currency2
    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    //On transforme les variables catégorielles ainsi crée en variables continues
    var encoderCountry = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_onehot")

    var encoderCurrency = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_onehot")


    //On assemble tous nos features dans une seule et même colonne
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

    //On applique la logistique regression à nos features
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

    //On assemble le tout dans un pipeline
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

    //Pour la crossvalidation, on génère le grid pour lequelle toutes les combinaisons seront testées
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.00000001,0.000001,0.0001,0.01))
      .addGrid(cvModel.minDF, Array(55.0,75.0,95.0))
      .build()

    //On se défini un evaluateur utilisant le score f1
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    //On défini notre fonction de crossvalidation basé sur le pipeline, le grid, et l'évaluator défini précédement
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    //Entrainement du modèle
    val cv_model =  trainValidationSplit.fit(training)

    //Prédiction sur le dataframe test
    val df_WithPredictions = cv_model.transform(test)

    //On calcul le f1 score sur les données prédites
    val f1 = evaluator.evaluate(df_WithPredictions)

    println("Score f1 du modèle après crossval : " + f1)

    //On affiche la matrice de confusion
    df_WithPredictions.groupBy("final_status", "predictions").count().show()

    //On sauvegarde enfin le modèle
    cv_model.write.overwrite().save("Cv_model")

  }
}
