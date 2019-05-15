import os
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from bs4 import BeautifulSoup
import numpy as np
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords 
import logging
import re
from gensim.models import word2vec
import numpy as np


def readData(base, count):
	
	dirlist = os.listdir(base)

	counter=0
	data={}

	for d in dirlist:
		
		d=str(d.encode('utf-8').strip()).strip("b\'").strip("\'")
		
		counter=counter+1
		if (counter==count):
			break

		path=base+"/"+d
		file=open(path, encoding="utf8", errors='ignore')
		index=0
		sections=[]
		
		for line in file:
			line=str(line.encode('utf-8').strip())
			index=index+1
			if(index%2==0):
				sections.append(line)
		
		data[d]=sections

	return data


def review_to_wordlist( review, remove_stopwords=False ):
   
	# 1. Remove HTML
	review_text = BeautifulSoup(review, "lxml").get_text()
	  
	# 2. Remove non-letters
	review_text = re.sub("[^a-zA-Z]"," ", review_text)

	# 3. Convert words to lower case and split them
	words = review_text.lower().split()

	# 4. Optionally remove stop words (false by default)
	if remove_stopwords:
		stops = set(stopwords.words("english"))
		words = [w for w in words if not w in stops]

	# 5. Return a list of words
	return(words)


# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):

	#Function to split a review into parsed sentences. 
	#Returns a list of sentences, where each sentence is 
	#a list of words

	#Split the paragraph into sentences
	#raw_sentences = tokenizer.tokenize(review.strip())
	#raw_sentences = tokenizer.tokenize(review)

	raw_sentences = re.split(r' *[\.\?!][\'"\)\]]* *', review)





	#Loop over each sentence
	sentences = []
	for raw_sentence in raw_sentences:
		# If a sentence is empty, skip it
		if len(raw_sentence) > 0:
			#Get a list of words
			sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))

	#Return the list of sentences 
	#Each sentence is a list of words,
	#so this returns a list of lists
	return sentences


def cleanData(data):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	return_list_sentences_for_each_section=[]

	for s in range(0,7):

		print ("s="+str(s))

		sentences=[]

		
		for d in data:
			
			section=data.get(d)[s]
			print ("d="+str(d)+"\tlen="+str(len(section)))


			sentences += review_to_sentences(section, tokenizer)		
		

		return_list_sentences_for_each_section.append(sentences)

	return return_list_sentences_for_each_section






# Function to average all of the word vectors in a given paragraph
def makeFeatureVec(words, model, num_features):
    
	# Pre-initialize an empty numpy array (for speed)
	featureVec = np.zeros((num_features,),dtype="float32")

	nwords = 0.
 
	#Index2word is a list that contains the names of the words in 
	#the model's vocabulary. Convert it to a set, for speed 
	#index2word_set = set(model.index2word)
	index2word_set = set(model.wv.index2word)

	# Loop over each word in the review and, if it is in the model's
	# vocaublary, add its feature vector to the total
	for word in words:
		if word in index2word_set: 
			nwords = nwords + 1.
			featureVec = np.add(featureVec,model[word])

	# Divide the result by the number of words to get the average
	featureVec = np.divide(featureVec,nwords)
	return featureVec


	# Given a set of reviews (each one a list of words), calculate 
	# the average feature vector for each one and return a 2D numpy array
def getAvgFeatureVecs(reviews, model, num_features):

	counter = 0

	# Preallocate a 2D numpy array, for speed
	reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    
	for review in reviews:

		if counter%1000 == 0:
			print( "Review %d of %d" % (counter, len(reviews)))

		reviewFeatureVecs[counter] = makeFeatureVec(review, model,num_features)

		counter = counter + 1

	return reviewFeatureVecs



if __name__ == '__main__':

	data=readData("./cleaned", 3)
	print (len(data))
	for d in data:

		print (data.get(d)[0])


	




	'''
	return_list_sentences_for_each_section=cleanData(data)

	# Set values for various parameters
	num_features = 300    # Word vector dimensionality                      
	min_word_count = 40   # Minimum word count                        
	num_workers = 4       # Number of threads to run in parallel
	context = 10          # Context window size                                                                                    
	downsampling = 1e-3   # Downsample setting for frequent words

	# Initialize and train the model (this will take some time)

	model = word2vec.Word2Vec(return_list_sentences_for_each_section[0], workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

	# If you don't plan to train the model any further, calling 
	# init_sims will make the model much more memory-efficient.
	model.init_sims(replace=True)

	# It can be helpful to create a meaningful model name and 
	# save the model for later use. You can load it later using Word2Vec.load()
#	model_name = "recommenderModel"
#	model.save(model_name)


	list_clean_sections=[]	
#	for s in range(0,7):
	for s in range(0,1):

		print ("s="+str(s))
		clean_sections = []
			
		for d in data:
				
			section=data.get(d)[s]
			print ("d="+str(d)+"\tlen="+str(len(section)))

			clean_sections.append( review_to_wordlist( section, remove_stopwords=True ))
		list_clean_sections.append(clean_sections)

	DataVecs = getAvgFeatureVecs( list_clean_sections[0], model, num_features )

	print (len(DataVecs[0]))
	print (len(DataVecs[1]))
	'''