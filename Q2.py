from pyspark.sql.types import StructField
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import functions as F




mismatches = spark.read.format("text").load("hdfs:///data/shared/msd/tasteprofile/mismatches/sid_mismatches.txt",header="TRUE")
mismatches.show(10,False)
mismatches.count()   #19094


matches=spark.read.format("text").load("hdfs:///data/shared/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt")
matches.show(10,False)
matches.count() #938

triplets_schema = StructType([StructField('User', StringType(), True),
                     StructField('SongID', StringType(), True),
                     StructField('Count', StringType(), True)                      
                     ])
 
triplets= spark.read.format("csv").schema(triplets_schema).option('delimiter','\t').load("hdfs:///data/shared/msd/tasteprofile/triplets.tsv",header="False")
triplets.count()  #48,373,586


triplets.select("SongID").distinct().count() #384546
triplets.select("User").distinct().count() #1019318

mismatches.select(mismatches.value.substr(9,18).alias('NewSongID')).show()


mismatches = mismatches.select(
        F.trim(F.substring(mismatches.value,  9, 18)).alias("SONGID").cast(StringType()),
        F.trim(F.substring(mismatches.value, 19, 28)).alias("TRACKID").cast(StringType()),
)
mismatches.show(10, False)

mismatch_final= (
  triplets.join(mismatches, triplets['SONGID'] == mismatches['SONGID'], 
    'leftanti')
  ) 

mismatch_final.show(10, False)

mismatch_final.count() #45,795,100 


#2b
attributes = (
     spark.read.format("csv").load("hdfs:///data/shared/msd/audio/attributes/")
)
attributes.show(10,False) 
attributes.select('_c0').distinct().show()

#Not helping

attributes = attributes.withColumnRenamed('_c0','Features').withColumnRenamed('_c1','Type')
sc_Attribute = attributes.schema

#Methodsofmoments turns out to be the smallest sized file with 37949102  bytes.

att_mom = (
     spark.read
      .format("csv")
      .option('delimiter',',')
      .schema(sc_Attribute)
      .load("hdfs:///data/shared/msd/audio/attributes/msd-jmir-methods-of-moments-all-v1.0.attributes.csv")
)

att_mom.show(50,False)

#Defining key value pairs
Dictionary = { 'string' : 'StringType()','STRING' : 'StringType()','NUMERIC': 'DecimalType(18,9)','real'   : 'DecimalType(18,9)' }

#Creating a new schema
new_schema = StructType([StructField(Features, eval(Dictionary[Type]), True) for (Features, Type) in att_mom.rdd.collect()])

att_momfeature = (
     spark.read
     .format("csv")
     .schema(new_schema)
     .option('delimiter',',')
     .load("hdfs:///data/shared/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv")
)


DescriptiveStatistics_sample1 = att_momfeature.describe()
DescriptiveStatistics_sample1.show()



#Audio Similarity 
#Import
aom = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/"))

jmir_lpc=(spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-jmir-lpc-all-v1.0.csv"))

mfcc = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv"))

marsyas = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-marsyas-timbral-v1.0.csv"))

spectralall = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv"))

spectral_der = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv"))

mvd = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-mvd-v1.0.csv"))

msd_rh = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-rh-v1.0.csv"))

msd_rp = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-rp-v1.0.csv"))

msd_ssd = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-ssd-v1.0.csv"))

trh = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-trh-v1.0.csv"))

tssd = (spark.read.format("csv").options(inferschema='true').load("hdfs:///data/shared/msd/audio/features/msd-tssd-v1.0.csv"))




aom.count() # 994623
jmir_lpc.count() # 994623
mfcc.count() # 994623
spectralall.count() # 994623
spectral_der.count() # 994623

marsyas.count() # 995001
mvd.count() # 994188
msd_rh.count() # 994188
msd_rp.count() # 994188
msd_ssd.count() # 994188
trh.count() # 994188
tssd.count() #994188


descstats= att_mom.describe()
descstats.show()


+-------+-------------------+
|summary|           Features|
+-------+-------------------+
|  count|             994623|
|   mean| 0.1549817600174646|
| stddev|0.06646213086143017|
|    min|                0.0|
|    max|          9.925e-06|
+-------+-------------------+



from pyspark.mllib.stat import Statistics
import pandas as pd

def compute_correlation_matrix(df, method='pearson'):
    df_rdd = df.rdd.map(lambda row: row[0:])
    corr_mat = Statistics.corr(df_rdd, method=method)
    corr_mat_df = pd.DataFrame(corr_mat,
                    columns=df.columns, 
                    index=df.columns)
    return corr_mat_df



corrmat = att_momfeature.drop('MSD_TRACKID')
corrmat = compute_correlation_matrix(corrmat)

#Creating a schema for all the MAGD files
sc_magd = StructType([
  StructField('id', StringType(), True), 
    StructField('genre', StringType(), True)]) 

#Importing MAGD
magd = (spark.read.format("csv")
  .schema(sc_magd)
    .option("delimiter", "\t")
    .load("hdfs:///data/shared/msd/genre/msd-topMAGD-genreAssignment.tsv"))

# Counts of each genre
magd.groupBy('genre').count().show()

#Importing other two datasets and thus loading the MSD all music genre dataset
magd_SA = (spark.read.format("csv").option('delimiter','\t').schema(sc_magd).load("hdfs:///data/shared/msd/genre/msd-MASD-styleAssignment.tsv"))
magd_SA.count()

magd_top = (spark.read.format("csv").option('delimiter','\t').schema(sc_magd).load("hdfs:///data/shared/msd/genre/msd-topMAGD-genreAssignment.tsv"))
magd_top.count()


Final_genre = (magd.join(magd_top.select(magd_top.genre.alias('TopGenre'),'id'), 'id', 'left')
  .join(magd_SA.select('id',magd_SA.genre.alias('Style')),'id','left'))


#F.col("TrackId"),
att_momfeature = att_momfeature.select(
    F.trim(F.regexp_replace(F.col('MSD_TRACKID'), "'", "")).alias("TrackId"),
    F.col("Method_of_Moments_Overall_Standard_Deviation_1").alias("SD1"),
    F.col("Method_of_Moments_Overall_Standard_Deviation_2").alias("SD2"),
    F.col("Method_of_Moments_Overall_Standard_Deviation_3").alias("SD3"),
    F.col("Method_of_Moments_Overall_Standard_Deviation_4").alias("SD4"),
    F.col("Method_of_Moments_Overall_Standard_Deviation_5").alias("SD5"),
    F.col("Method_of_Moments_Overall_Average_1").alias("A1"),
    F.col("Method_of_Moments_Overall_Average_2").alias("A2"),
    F.col("Method_of_Moments_Overall_Average_3").alias("A3"),
    F.col("Method_of_Moments_Overall_Average_4").alias("A4"),
    F.col("Method_of_Moments_Overall_Average_5").alias("A5")
    )

Final_genre = Final_genre.withColumnRenamed("id", "TrackID")

Finalnew = att_momfeature.join(Final_genre, on="TrackID", how='inner')                  

Merge_audio=Finalnew.drop('SD2','SD3','SD4','A3','A4', 'Style')


def is_electronic(genre):
    answer = 0
    if genre == 'Electronic':
        answer = 1
    return answer

is_electronic_udf = F.udf(is_electronic, IntegerType())

data = Merge_audio.withColumn("label", is_electronic_udf(Merge_audio.TopGenre))
#data = datanew.drop("genre")

#-> What is the class imbalance?
data.filter(data["label"] == 1).count() #40666

data.count() #404395

#Class imbalanace = 40666/404395 - 10%.
#############################################################################
## Converting the Dataframe to list 
newdata=data.collect()

#To match majprity class
from sklearn.utils import resample

resampledata = resample(newdata, 
                                 replace=True,     
                                 n_samples=379954,    
                                 random_state=123)


### Converting the list to Datframe
classbalance = spark.createDataFrame(resampledata)



## Dividing the Data File as The majority and Minority Datasets 
majority_class=Merge_audio.filter(datanew.label==0)
minority_class=Merge_audio.filter(datanew.label==1)



new_resampled_data = majority_class.union(classbalance)
########################################################################
fraction = {0: 0.5, 1: 0.5}
new_resampled_data = data.sampleBy("label", fractions=fraction , seed=1)
#######################################################################

#Modelling
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
renamed_data=new_resampled_data
renamed_data = renamed_data.withColumn("label", renamed_data["label"].cast(IntegerType()))


assembler = VectorAssembler().setInputCols(("SD1","SD5","A1","A2","A5")).setOutputCol("features")

renamed_data = assembler.transform(renamed_data)

(trainingData, testingData) = renamed_data.randomSplit([0.75, 0.25], seed = 5)

from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression().setMaxIter(20).setRegParam(0.3).setFeaturesCol("features").setLabelCol("label")

from pyspark.ml import Pipeline
#pipeline_lr = Pipeline().setStages((assembler,lr))
log_model = lr.fit(trainingData) 
pred = log_model.transform(testingData)


# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(log_model.coefficients))
print("Intercept: " + str(log_model.intercept))


#Computing range of metrics for each of the algorithms
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
auroc = binary_evaluator.evaluate(pred)
print('auroc: {}'.format(auroc))

#auroc: 0.671

pred_sub = pred.select("label", "prediction")

tn = pred_sub.filter((pred_sub.label==0) & (pred_sub.prediction==0)).count()
tp = pred_sub.filter((pred_sub.label==1) & (pred_sub.prediction==1)).count()
fp = pred_sub.filter((pred_sub.label==0) & (pred_sub.prediction==1)).count()
fn = pred_sub.filter((pred_sub.label==1) & (pred_sub.prediction==0)).count()

precision = tp / (tp + fp) 
accuracy = (tp + tn) / (tn+tp+fp+fn) #0.89 
recall = tp / (tp + fn) 




#Random Forest

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees = 100, maxDepth = 4, maxBins = 32) 
rfModel = rf.fit(trainingData)
predictions_rf = rfModel.transform(testingData)
evaluator_rf_accuracy = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
evaluator_rf_Recall = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("weightedRecall")
evaluator_rf_Precision = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("weightedPrecision")
evaluator_rf_Precision.evaluate(predictions_rf) #0.8045
evaluator_rf_Recall.evaluate(predictions_rf) #0.8969
evaluator_rf_accuracy.evaluate(predictions_rf) #0.8969


#Decision TRee
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 30)
dtModel = dt.fit(trainingData)
predictions_dt = dtModel.transform(testingData)
evaluator_dt_accuracy = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
evaluator_dt_recall = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("weightedRecall")
evaluator_dt_precision = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("weightedPrecision")
evaluator_dt_precision.evaluate(predictions_dt) #0.8441
evaluator_dt_recall.evaluate(predictions_dt)  ##0.8416
evaluator_dt_accuracy.evaluate(predictions_dt)  ##0.8415


#q3
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
Merge_audio=Finalnew.drop('SD2','SD3','SD4','A3','A4')


from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

label_stringIdx = StringIndexer(inputCol = "TopGenre", outputCol = "indexlabel")
pipeline = Pipeline(stages=[label_stringIdx])
pipelineFit = pipeline.fit(Merge_audio)
dataset = pipelineFit.transform(Merge_audio)

dataset = dataset.withColumn("label", dataset["indexlabel"].cast(IntegerType()))
dataset=dataset.drop("indexlabel")
assembler = VectorAssembler().setInputCols(("SD1","SD5","A1","A2","A5")).setOutputCol("features")


Merge_audio=Merge_audio.filter(Merge_audio.genre.isNotNull())

sampled=assembler.transform(dataset)
(train, test) = sampled.randomSplit([0.7, 0.3],seed=100)
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
ovr = OneVsRest(classifier=lr)
ovrModel = ovr.fit(train)
predictions = ovrModel.transform(testingData)

# Evaluating the Values

