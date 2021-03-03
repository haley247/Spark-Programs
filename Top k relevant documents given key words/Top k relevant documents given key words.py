#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql.session import SparkSession
from os import listdir
import re
import math
import string

conf = SparkConf()
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

with open('stopwords.txt','r') as f:
    stopwords = f.read().splitlines() 
    
datafiles = listdir('datafiles')
datafiles = [filename for filename in datafiles if filename.endswith('txt')]
n = len(datafiles)

#Step 1. Preprocess the documents. (Refer to Data Pre-processing section) 

def process_single(line): 
    words = line.split() #strip line into words
    words = [re.sub('\W+','', w.lower()) for w in words] #strip out special symbols, ending symbols and convert to lowercase
    words = [w for w in words if not w.isdigit()] #drop independent numbers (not alphanumeric) in sentences
    return [w for w in words if w not in stopwords] #check and remove stopwords

#Step 2: Compute term frequency (TF) of every word in a document.
# TF is the count of the word in a document

tf = sc.parallelize([])
n = len(datafiles)
for i in range(n):
    lines = sc.textFile("datafiles/%s" % (datafiles[i]))
    words = lines.flatMap(process_single)
    words = words.map(lambda w: ((w, i), 1)).reduceByKey(lambda a, b: a + b)
    tf = sc.union([tf, words])

#DF is the count of documents having the word
df = tf.map(lambda x: (x[0][0], 1)).reduceByKey(lambda a,b: a+b)  

#Step 3. Compute TF-IDF of every word w.r.t a document.
#You can use key-value pair RDD and the groupByKey() API. Use log base 10 for TF-IDF calculation.
# TF-IDF = (1 + log (TF)) * log (N/DF)

def tf_idf_cal(word):
    tfidf = (1+math.log(word[1][0]))*math.log10(n/word[1][1])
    return (word[0], tfidf)

#Calculating query value
lines = sc.textFile("query.txt")
words = lines.flatMap(process_single)
query_rdd = words.map(lambda w: (w, 1)).reduceByKey(lambda n1, n2: n1 + n2)
query_to_filter = query_rdd.keys().collect()
query_score = math.sqrt(query_rdd.reduce(lambda x,y: x[1]**2+y[1]**2))

#Step 4. Compute normalized TF-IDF of every word w.r.t. a document.
#Step 5. Compute the relevance of each document w.r.t a query.
relevance_score = sc.parallelize([])
for i in range(n):
    df_final = df.map(lambda word: ((word[0],i),word[1]))
    tfidf_cal = tf.join(df_final).map(tf_idf_cal)
    S_score = math.sqrt(tfidf_cal.map(lambda word: (word[0][1], word[1])).values().map(lambda word: word*word).sum())
    normalized_tfidf = tfidf_cal.map(lambda word: (word[0], word[1]/S_score))
    
    match1 = normalized_tfidf.filter(lambda word: word[0][0] in query_to_filter).map(lambda word: (word[0][0], word[1]))
    match2 = match1.join(query_rdd).map(lambda word: (word[0],word[1][0]*word[1][1])).values().sum()
   
    match3 = normalized_tfidf.values().map(lambda word: word**2).sum()
    relevance = sc.parallelize([(datafiles[i], match2/match3/query_score)])
    relevance_score = relevance_score.union(relevance)

B_Top5 = relevance_score.sortBy(lambda x:x[1],ascending=False).take(5)

#Step 6. Sort and get top-k documents.
with open('TaskB.txt','w') as f:
        for line in B_Top5:
                f.write(str(line[0])+', '+str(line[1]))
                f.write('\n')
sc.stop()


# In[ ]:




