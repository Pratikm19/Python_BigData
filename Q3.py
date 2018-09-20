triplets_schema = StructType([StructField('User', StringType(), True),
                     StructField('SongID', StringType(), True),
                     StructField('Count', StringType(), True)                      
                     ])
 
triplets= spark.read.format("csv").schema(triplets_schema).option('delimiter','\t').load("hdfs:///data/shared/msd/tasteprofile/triplets.tsv",header="False")
triplets.count() 


#The triplets dataset comprises of three columns User, Song ID, Count
triplets.select('User').distinct().count() #Number of Unique users - 1019318
triplets.select('SongID').distinct().count() #Number of Unique songs - 384546



from pyspark.sql.functions import count, sum, avg, max
import pyspark.sql.functions as F
#Count Distincts
triplets.groupBy("User").agg(countDistinct(F.col("SongID")).alias("new")).orderBy(F.col("new").desc()).show()


triplets.select('SongID').distinct('ec6dfcf19485cb011e0b22637075037aae34cf26').count()


triplets.filter(triplets["User"]=='ec6dfcf19485cb011e0b22637075037aae34cf26').select("SongID").distinct().count()
#4400

#Visualise the distribution of song popularity
triplets.write.csv("hdfs:///home/pme67/data/outputs/songreccomendation.csv")
hdfs dfs -copyToLocal /home/pme67/data/outputs/triplets /users/home/pme67

hdfs dfs -copyToLocal /home/pme67/data/outputs/songreccomendation.csv /users/home/pme67

#d
new_song=triplets.groupBy("SongID").agg(sum("Count").alias("newcolumn"))
new_user=triplets.groupBy("User").agg(sum("Count").alias("newcolumn"))

new_song.orderBy(F.col('newcolumn').desc()).show(10)

new_user.orderBy(F.col('newcolumn').asc()).show(10) #User has heard 10 songs minimum 
 
new_song.orderBy(F.col('newcolumn').asc()).show(10) #1 song is played each time, so 

#Triplets filtering

new_triplets=triplets.filter(triplets["Count"]>5) #An individual could have listened to 5 different songs at the most, so i have chosen random 
new_triplets.orderBy(F.col("Count").asc()).show(10)  		
new_triplets.orderBy(F.col("Count").desc()).show(10)


new_user2=new_triplets.groupBy("User").agg(sum("Count").alias("newcolumn"))
new_user2.orderBy(F.col("newcolumn").asc()).show()
new_triplets= new_triplets.filter(triplets["Count"]>8) #Greater than 8 considered

#Splitting the data

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

train, test = new_triplets.randomSplit([0.8, 0.2], seed=12345)

#Q2a

from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
rank = 10
numIterations = 10
als = ALS(maxIter=5, regParam=0.01, userCol="Count", itemCol="User", ratingCol="SongID",
          coldStartStrategy="drop")

model = als.fit(train)









