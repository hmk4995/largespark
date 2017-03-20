# import section
import os
from pyspark import SparkContext
import nltk
import string
import codecs
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import log10
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
#import feature_selection as fs

# globals

data_folder = '/home/harikrishnan/data'
stop = stopwords.words('english')
words = []
punct = string.punctuation
stemmer = PorterStemmer()
sc = SparkContext('local', 'project')
sc.setCheckpointDir('hdfs://master:9000/RddCheckPoint')
voc = []

#functions

#Reading the corpus as an RDD

def read_data( data_folder  ):
	data = sc.parallelize([])
	for folder in os.listdir( data_folder  ):
   		for txt_file in os.listdir( data_folder + '/' + folder  ):
   			temp = open( data_folder + '/' + folder + '/' + txt_file)
   			temp_da = temp.read()
   			temp_da = unicode(temp_da, errors = 'ignore')
   			temp.close()
   			a = [ ( folder, temp_da) ]
			data = data.union(sc.parallelize( a ) )
	return data
    		
    			
# Preprocessing the documents 

def preprocessing( text ):
	tokens = word_tokenize(text)
	lowercased = [t.lower() for t in tokens]
	rem_stp_punct = [ w for w in lowercased if w not in stop and w not in punct]
	#final = [ stemmer.stem(w) for w in rem_stp_punct]
	processed_txt = " ".join(rem_stp_punct)
	return processed_txt
	

# Converting the preprocessed RDD to a form thats suitable for feature weighting
def transform_for_idf(words):
	
	for elmnt in words:
		yield (elmnt, 1)

# Calculating tf values of terms in a document		
def get_tf( doc):
	temp_doc = []
	uniq = set( doc)
	for w in uniq:
		temp_doc.append(( w, doc.count( w)))
	for w in voc:
		if w not in uniq:
			temp_doc.append((w, 0))
	return sorted(temp_doc)
		
# Calculating idf values of terms in a document
def get_tf_idf( doc):
	temp_doc = []
	for ( feat, tf) in doc:
		temp_doc.append( (feat, tf * data_for_idf[ feat]))
	return temp_doc

#Changing Class name into double value for LabeledPoint object
def class_number(x):
	return { 
		'civ': 0.0,
		'CS' : 1.0,
		'ece': 2.0,
	       }[x]

#Changing int LabeledPoint type before classification
def pre_svm_represent(label,doc):
	temp_vector=[]
	for(feat, tfidf) in doc:
		temp_vector.append(tfidf)
	return LabeledPoint(class_number(label),temp_vector)

#######################################################################
# Converting the preprocessed RDD to a form thats suitable for feature selection

def transform_for_feat_slct( label, words):
	
	for elmnt in words:
		yield (elmnt, ( label, 1))


# Calculating CHI square value of a term

def cal_chi( all_freq, pos_freq, pos_num, total_num):
    pos_freq = pos_freq if pos_freq else 0
    numertr = 1.0 * total_num * ( ( pos_freq * total_num - all_freq * pos_num) ** 2) 
    denom = pos_num * pos_freq * ( total_num - pos_num) * ( total_num - all_freq)
    if denom == 0:
    	return 0
    else:
    	return numertr / denom




# Calculating CHI square value of an RDD representing documents in a class

def get_vals( folder , feat_cls_count, feat_sel_data, feats_freq_all, total_num):
	feats = feat_cls_count.filter( lambda ( f, ( label, freq)) : label == folder)
	feats_freq_pos = feats.reduceByKey( lambda a, b : (a[0], a[1] + b[1])).mapValues( lambda ( label, freq) : freq)
	feat_pos_num = feat_sel_data.filter( lambda ( label, text) : label == folder).count()
	union = feats_freq_all.fullOuterJoin( feats_freq_pos)
	chi_rdd = union.mapValues( lambda ( all_freq, pos_freq) : cal_chi( all_freq, pos_freq, feat_pos_num, total_num))
	return chi_rdd
	
#######################################################################

   
#################################### main #############################
  
#Getting data as RDD and preprocessing  	
inpt_data = read_data( data_folder )
preprocessed_data = inpt_data.map( lambda ( label, text ) :  ( label, preprocessing( text )))

#print preprocessed_data.take(2)


#Feature selection
feat_sel_data = preprocessed_data.map( lambda ( label, text ) : (label, list( set(text.split()))))
#print feat_sel_data.take(2)
print "ah"
feat_cls_count = feat_sel_data.flatMap( lambda ( label, words) : transform_for_feat_slct( label, words))
#print feat_cls_count.take(2)
print "aho"
feats_freq_all = feat_cls_count.map( lambda ( f, ( label, freq)): (f, freq)).reduceByKey(lambda a, b: a + b)
#print feats_freq_all.take(2)
print "ahoy"
total_num = inpt_data.count()
print total_num
print "ahoye"

civ_rdd_chi = get_vals('civ', feat_cls_count, feat_sel_data, feats_freq_all, total_num)
#print civ_rdd_chi.take(2)
cs_rdd_chi = get_vals('CS', feat_cls_count, feat_sel_data, feats_freq_all, total_num)
#ece_rdd_chi = get_vals('ece',feat_cls_count, feat_sel_data, feats_freq_all, total_num)
temp_chi_data = civ_rdd_chi.union( cs_rdd_chi )
#temp_chi_data = temp_chi_data.union( ece_rdd_chi )
chi_data = temp_chi_data.reduceByKey(max)
sort_chi_data = chi_data.sortBy( lambda x: x[1], False)
vocabulary = sort_chi_data.map( lambda ( feature, chi) : feature).take(50)
voc = sorted([ w for w in vocabulary])
#print voc


#Feature weighting and representing documents

data_for_tf = preprocessed_data.map( lambda ( label, text) : (label, text.split()))
#print data_for_tf.take(2)
#n = data_for_tf.count()
#print n
data_for_tf = data_for_tf.map( lambda (label, doc) : (label, [w for w in doc if w in voc]))
#print data_for_tf.take(2)
print "hi"
data_for_idf = data_for_tf.map( lambda (label, doc) : (label, list( set( doc))))
#print data_for_idf.take(2)
print "hel"
data_for_idf = data_for_idf.flatMap( lambda (label, doc) :  transform_for_idf( doc))
#print data_for_idf.take(2)
print "hell"
data_for_idf = data_for_idf.reduceByKey( lambda a, b: a + b)
#print data_for_idf.take(2)

data_for_idf = data_for_idf.map( lambda ( feat, df): ( feat, log10( (total_num + 1.0) / df)))
#print data_for_idf.take(10)
data_for_idf = data_for_idf.sortBy( lambda x : x[0]).collectAsMap()
#print type(data_for_idf)
#print data_for_idf.take(2)
#data_for_idf.count()
#it = data_for_idf.toLocalIterator()
#print type(it)
#data_for_idf = dict([ idf for idf in it ])
#print data_for_idf
#print type(data_for_idf)
data_for_tf = data_for_tf.map( lambda (label, doc) : (label, get_tf(doc)))
print "hello"
#print data_for_tf.take(2)
data_for_tfidf = data_for_tf.map( lambda (label,doc) : (label, get_tf_idf( doc)))
print "helloy"
#print data_for_tfidf.take(2)
pre_svm_data = data_for_tfidf.map(lambda (label, doc) : pre_svm_represent(label,doc))
print pre_svm_data.take(4)

