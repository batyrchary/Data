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



def review_to_words( raw_review ):

	# 1. Remove HTML
	review_text = BeautifulSoup(raw_review, "lxml").get_text() 
	
	# 2. Remove non-letters        
	letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
	
	# 3. Convert to lower case, split into individual words
	words = letters_only.lower().split()                             
	
	# 4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
	stops = set(stopwords.words("english"))                  
	 
	# 5. Remove stop words
	meaningful_words = [w for w in words if not w in stops]   
	
	# 6. Join the words back into one string separated by space, and return the result.
	return( " ".join( meaningful_words )) 


def cleanData(data):

	clean_papers = {}

	for paper in data:
		rawSections=data.get(paper)
		cleanSections=[]

		for s in rawSections:
			cleanSections.append(review_to_words(s))

		clean_papers[paper]=cleanSections

	return clean_papers
	

def bagOfWord(data):
	
	vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None, stop_words = None, max_features = 10) 

	#fit_transform() 
	#First, it fits the model and learns the vocabulary; 
	#Second, it transforms our training data into feature vectors.
	#The input to fit_transform should be a list of strings.


	data_features_dic={}


	for s in range(0,7):

		sections=[]
		for d in data:
			sections.append(data.get(d)[s])
			
		section_features = vectorizer.fit_transform(sections)
		np.asarray(section_features)

		counter=0
		for d in data:
			sf=section_features[counter]
			if d in data_features_dic:
				old=data_features_dic.get(d)
				old.append(sf)
				data_features_dic[d]=old
			else:
				data_features_dic[d]=[sf]
			counter=counter+1

	return data_features_dic		

#	Sentence 1: "The cat sat on the hat"

#	Sentence 2: "The dog ate the cat and the hat"

#	vocabulary is as follows: -> { the, cat, sat, on, hat, dog, ate, and }

#	Sentence 1: { 2, 1, 1, 1, 1, 0, 0, 0 }

#	Sentence 2 are: { 3, 1, 0, 0, 1, 1, 1, 1}


if __name__ == '__main__':

	data=readData("./cleaned", 3)
	print (len(data))

	clean_data=cleanData(data)

	data_features_dic=bagOfWord(clean_data)

	print (len(data_features_dic))
	for d in data_features_dic:
		sf=data_features_dic.get(d)

		for s in sf:
			print (s)
			print ("---------")

		print ("<<<<<<<<<<<>>>>>>>>>>>>>>")


	



	


			