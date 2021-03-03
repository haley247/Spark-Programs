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

def process_pair(line): 
    words = line.split() #strip line into words
    #words = words.remove('')
    words = [re.sub('\W+','', w.lower()) for w in words] #strip out special symbols, ending symbols and convert to lowercase
    words = [w for w in words if not w.isdigit()] #drop independent numbers (not alphanumeric) in sentences
    words = [w for w in words if w not in stopwords] #check and remove stopwords
    return [sorted((words[i],words[i+1]))for i in range(0,len(words)-1)] #sort pair so that <word1, word2> and <word2, word1> are considered as the same

#Step 2. Compute the count of every word pair in the resulting documents. Note that
allpair = sc.parallelize([])

for i in range(n):
    text_file = sc.textFile("datafiles/%s" % (datafiles[i]))
    pairs = text_file.flatMap(process_pair)
    count = pairs.map(lambda x: ((x[0], x[1]), 1)).reduceByKey(lambda a, b: a + b)
    allpair = sc.union([allpair,count])

#Step 3. Sort the list of word pairs in descending order and obtain the top-k frequently occurring word pairs
A_Top5 = allpair.reduceByKey(lambda a, b: a + b).filter(lambda x: x[0] != ('', '')).sortBy(lambda x:x[1],ascending=False).take(5)

with open('TaskA.txt','w') as f:
        for line in A_Top5:
                f.write(str(line[0])+', '+str(line[1]))
                f.write('\n')
sc.stop()


# In[ ]:




