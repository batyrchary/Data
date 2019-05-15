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
from gensim.models import doc2vec
from collections import namedtuple
from gensim.models import doc2vec
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy
import math




def my_cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)


def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def pearson_def(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def jaccard_similarity(xu,yu):

	#print (xu)
	#print (yu)

	x = [ '%.1f' % elem for elem in xu ]
	y = [ '%.1f' % elem for elem in yu ]

	#print (x)
	#print (y)

	intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
	union_cardinality = len(set.union(*[set(x), set(y)]))
	return intersection_cardinality/float(union_cardinality)
  


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
			word_list=review_to_wordlist( raw_sentence, remove_stopwords )
			sentences.append(' '.join(word_list))

	#Return the list of sentences 
	#Each sentence is a list of words,
	#so this returns a list of lists
	return sentences



def cleanDataU(data):
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	return_dic_of_papers_with_each_section_list={}

	for d in data:
		list_of_each_section=[]

		for s in range(0,7):
			section=data.get(d)[s]

			sentences = review_to_sentences(section, tokenizer)

			list_of_each_section.append('. '.join(sentences))

		return_dic_of_papers_with_each_section_list[d]=list_of_each_section

	return return_dic_of_papers_with_each_section_list	











if __name__ == '__main__':

	data=readData("./cleaned", 4)
	#print (len(data))
	#for d in data:
	#	print (data.get(d)[0])


	
	return_dic_of_papers_with_each_section_list=cleanDataU(data)

#	print(len(return_dic_of_papers_with_each_section_list))	
#	for d in return_dic_of_papers_with_each_section_list:
#		print (len(return_dic_of_papers_with_each_section_list.get(d)))
#		print (len(return_dic_of_papers_with_each_section_list.get(d)[0]))
#		print (len(return_dic_of_papers_with_each_section_list.get(d)[0][0]))
#		print (len(return_dic_of_papers_with_each_section_list.get(d)[1]))


	vectors_paper_listofvectors={}
	for d in return_dic_of_papers_with_each_section_list:
		#print (d)
		vectors_paper_listofvectors[d]=[]

	for d in vectors_paper_listofvectors:
			mlist=vectors_paper_listofvectors.get(d)
			#print (mlist)



	for s in range(0,7):
	
		print ("s="+str(s))

		doc=[]
		for d in return_dic_of_papers_with_each_section_list:
			section=return_dic_of_papers_with_each_section_list.get(d)[s]
			doc.append(section)

		docs = []
		analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

		for i, text in enumerate(doc):
			words = text.lower().split()
			tags = [i]
			docs.append(analyzedDocument(words, tags))

		alpha_val = 0.025        # Initial learning rate
		min_alpha_val = 1e-4     # Minimum for linear learning rate decay
		passes = 10              # Number of passes of one document during training

		print("letsbuild")


		alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)
		model = doc2vec.Doc2Vec( vector_size = 150, window = 150, min_count = 1, workers = 4)		
		
		model.build_vocab(docs) # Building vocabulary

		for epoch in range(passes):

			# Shuffling gets better results	    
			random.shuffle(docs)

			# Train
			model.alpha, model.min_alpha = alpha_val, alpha_val
			model.train(docs,total_examples=model.corpus_count,epochs=model.epochs)

			# Logs
			#print('Completed pass %i at alpha %f' % (epoch + 1, alpha_val))

			# Next run alpha
			alpha_val -= alpha_delta

		print("finishedbuild")
		counter=0
		for d in vectors_paper_listofvectors:
			mlist=vectors_paper_listofvectors.get(d)
			#print (len(mlist))
			mlist.append(model.docvecs[counter])
			vectors_paper_listofvectors[d]=mlist
			counter=counter+1



#		print(my_cosine_similarity(model.docvecs[0],model.docvecs[1]))	
#		print(pearson_def(model.docvecs[0],model.docvecs[1]))
#		print(jaccard_similarity(model.docvecs[0],model.docvecs[1]))
	
	similarity_dic={}	
	for d1 in vectors_paper_listofvectors:
		
		
		for d2 in vectors_paper_listofvectors:
			list_of_similarities=[]

			#print(str(d1)+","+str(d2))
			if ((d1,d2) in similarity_dic) or ((d2,d1) in similarity_dic) or (d1==d2):
				continue

			#print(str(d1)+","+str(d2))
			#print("---------------")

			list1=vectors_paper_listofvectors.get(d1)
			list2=vectors_paper_listofvectors.get(d2)


			for i in range(0,len(list1)):

				cosine=my_cosine_similarity(list1[i],list2[i])	
				pearson=pearson_def(list1[i],list2[i])
				jaccard=jaccard_similarity(list1[i],list2[i])
				list_of_similarities.append([cosine,pearson,jaccard])

			if len(list_of_similarities)!=0:
				similarity_dic[(d1,d2)]=list_of_similarities

	printline="paper1,paper2,abstract_c,abstract_p,abstract_j,intro_c,intro_p,intro_j,related_c,related_p,related_j"
	printline=printline+",meth_c,meth_p,meth_j,eval_c,eval_p,eval_j,conc_c,conc_p,conc_j,rest_c,rest_p,rest_j"

	file=open("output.csv","w+")

	print(printline)

	file.write(printline+"\n")


	for dtuple in similarity_dic:
		ls=similarity_dic.get(dtuple)
	
		printline=str(dtuple[0])+","+str(dtuple[1])
		printline=printline+","+str(ls[0][0])+","+str(ls[0][1])+","+str(ls[0][2])
		printline=printline+","+str(ls[1][0])+","+str(ls[1][1])+","+str(ls[1][2])
		printline=printline+","+str(ls[2][0])+","+str(ls[2][1])+","+str(ls[2][2])
		printline=printline+","+str(ls[3][0])+","+str(ls[3][1])+","+str(ls[3][2])
		printline=printline+","+str(ls[4][0])+","+str(ls[4][1])+","+str(ls[4][2])
		printline=printline+","+str(ls[5][0])+","+str(ls[5][1])+","+str(ls[5][2])
		printline=printline+","+str(ls[6][0])+","+str(ls[6][1])+","+str(ls[6][2])


		print (printline)
		file.write(printline+"\n")

#		print (str(dtuple[0])+","+str(dtuple[1])+","+ls[0]+","+ls[1]+","+ls[2])
#		print (similarity_dic.get(dtuple))

	file.close()
		
#		print (len (vectors_paper_listofvectors.get(d)))




	'''
	# Load data
	doc1 = ["This is a sentence", "This is another sentence"] 
	docs = []
	analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')

	for i, text in enumerate(doc1):
		words = text.lower().split()
		tags = [i]
		docs.append(analyzedDocument(words, tags))


	
	# Train model (set min_count = 1, if you want the model to work with the provided example data set)
	#model = doc2vec.Doc2Vec(docs, vector_size = 10, window = 300, min_count = 1, workers = 4)

	# Get the vectors
	#print(model.docvecs[0])
	#print(model.docvecs[1])
	#print(len(model.docvecs))
	

	alpha_val = 0.025        # Initial learning rate
	min_alpha_val = 1e-4     # Minimum for linear learning rate decay
	passes = 15              # Number of passes of one document during training

	alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)
	model = doc2vec.Doc2Vec( vector_size = 2, window = 1, min_count = 1, workers = 4)
	model.build_vocab(docs) # Building vocabulary

	for epoch in range(passes):

		# Shuffling gets better results	    
		random.shuffle(docs)

		# Train
		model.alpha, model.min_alpha = alpha_val, alpha_val
		model.train(docs,total_examples=model.corpus_count,epochs=model.epochs)

		# Logs
		print('Completed pass %i at alpha %f' % (epoch + 1, alpha_val))

		# Next run alpha
		alpha_val -= alpha_delta

#	print(len(model.docvecs))
#	print(model.docvecs[0])
#	print(model.docvecs[1])
	

#	print(my_cosine_similarity(model.docvecs[0],model.docvecs[1]))
	#print(cosine_similarity([v1,v2]))
#	print(numpy.corrcoef(model.docvecs[0],model.docvecs[1])[0, 1])
#	print(pearson_def(model.docvecs[0],model.docvecs[1]))
#	print(jaccard_similarity(model.docvecs[0],model.docvecs[1]))

	'''


