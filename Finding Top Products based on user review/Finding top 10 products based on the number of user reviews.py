#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession

import gzip
import json

conf = SparkConf()
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

# Convert to 'strict' json
def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.dumps(eval(l))

# Open Review Data
f = open("output.strict", 'w')
for l in parse("reviews_Digital_Music.json.gz"):
  f.write(l + '\n')

df1 = spark.read.json("output.strict")



# Open Metadata
f = open("meta.strict", 'w')
for l in parse("meta_Digital_Music.json.gz"):
  f.write(l + '\n')

df2 = spark.read.json("meta.strict")

#Step 1: Find the number of unique reviewer IDs for each product from the review file.
#Handle Review Data

review = df1.select('asin','reviewerID').rdd.map(tuple)
new_review = review.map(lambda x: ((x[0], x[1]), 1))
new_review = new_review.reduceByKey(lambda a, b: a + b)
new_review_grouped = new_review.map(lambda w: (w[0][0], 1))
rdd1 = new_review_grouped.reduceByKey(lambda a, b: a + b)

#Step 2: Create an RDD, based on the metadata

meta = df2.select('asin','price').filter('price is not null').rdd.map(tuple)
rdd2 = meta.reduceByKey(lambda a, b: mean(a,b))

#Step 3: Join the pair RDD

joined = rdd1.join(rdd2)
#Sort to take top 10
sorted_rdd = joined.sortBy(lambda x: -x[1][0])

#Step 4: Display the product ID

top10 = sorted_rdd.zipWithIndex().filter(lambda vi: vi[1] < 10)
#Save Output
top10.map(lambda x:[x[0][0], x[0][1][0],x[0][1][1]]).repartition(1).saveAsTextFile('./output.txt')

sc.stop()





