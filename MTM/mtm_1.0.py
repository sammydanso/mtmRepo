# coding: utf-8
import csv
import re
import os
import csv
import glob
import codecs
import sys
import ntpath
import pandas as pd
from pandas import ExcelFile
import nltk
from sklearn import preprocessing
import nltk
from sklearn.externals import joblib
import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from collections import Counter
from nltk.stem import PorterStemmer
import sklearn

#nltk.download('punkt')

from nltk.tokenize import word_tokenize

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn import metrics


datadir = os.path.join(os.getcwd(), "corpus")

modeldir = os.path.join(os.getcwd(), "model")

outputDir = os.path.join(os.getcwd(), "output")


# KW list to be used as features - extracted using corupus linguistics techniques

data = pd.read_csv(os.path.join(datadir, "sample.csv"))
data.applymap(lambda s:s.lower() if type(s) == str else s)


d_and_aFeatures = pd.read_csv(os.path.join(datadir, "d_and_a_keywords.csv"))
d_and_aFeatures.applymap(lambda s:s.lower() if type(s) == str else s)

adjustmentFeatures = pd.read_csv(os.path.join(datadir, "adjustment_keywords.csv"))
adjustmentFeatures.applymap(lambda s:s.lower() if type(s) == str else s)


otherFeatures = pd.read_csv(os.path.join(datadir, "other_keywords.csv"))
otherFeatures.applymap(lambda s:s.lower() if type(s) == str else s)

features = pd.concat([d_and_aFeatures, adjustmentFeatures, otherFeatures])


d_and_a = data[data['class'] =='d_and_a']
adjustment = data[data['class'] =='adjustment']
other = data[data['class'] =='other']


def creatFeatureVector():
		ps = PorterStemmer()
		document_class = []
		dict_d_and_a = d_and_aFeatures.to_dict()
		dict_adjustment = adjustmentFeatures.to_dict()
		dict_other = otherFeatures.to_dict()
		stemmed_d_and_a = []
		stemmed_adjustment = []
		stemmed_other = []
		for t in dict_d_and_a:
			stemmed_d_and_a.append(ps.stem(t))
		for t in dict_adjustment:
			stemmed_adjustment.append(ps.stem(t))
		for t in dict_other:
			stemmed_other.append(ps.stem(t))
					
		for document in data['Document']:
			d_and_a_count =[]
			adjustment_count = []
			other_count = []
			tokens = nltk.word_tokenize(document)
			featureVector = {}
			for token in tokens:
				if ps.stem(token) in stemmed_d_and_a:
					
					d_and_a_count.append(ps.stem(token))
				if ps.stem(token) in stemmed_adjustment:
					adjustment_count.append(ps.stem(token))
				if ps.stem(token) in stemmed_other:
					other_count.append(ps.stem(token))
					
			d_and_a_counter=Counter(d_and_a_count)
			d_and_a_counter['label'] = 'd_and_a'
			adjustment_count_counter=Counter(adjustment_count)
			adjustment_count_counter['label'] = 'adjustment'
			other_count_counter=Counter(other_count)
			other_count_counter['label'] = 'other'
			document_class.append(d_and_a_counter)
			document_class.append(adjustment_count_counter)
			document_class.append(other_count_counter)
			
		dataTransformed = pd.DataFrame(document_class).fillna(0)
				
		#dataTransformed.to_excel(os.path.join(outputDir, 'trainingData.xlsx'))
		return dataTransformed
			

def buildModel(data):
	trainingData = data
	trainingSet_label = trainingData.label 
	trainingSet_data = trainingData.drop('label', axis= 1)
	X_train, X_test, y_train, y_test = train_test_split(trainingSet_data, trainingSet_label)
	scv = StratifiedKFold(n_splits=10)
	params = {}
	NB = GaussianNB()
	model = GridSearchCV(NB, cv=scv, param_grid=params, scoring = "accuracy", return_train_score=True)
	model.fit(X_train, y_train)
	print(model)
	print(model.score(X_test, y_test))
		
	# save the model to disk
	with open(os.path.join(modeldir, 'NBModel.pkl'),'wb') as savedModel:
		pickle.dump(model, savedModel)
	return model

def loadModel ():
	with open(os.path.join(modeldir, 'NBModel.pkl'),'rb') as savedModel:
		model = pickle.load(savedModel)
	return model
    
def predictCategory(inputText):
	 x = inputText
	 modelNB = loadModel()
	# print(x.shape)
	 y = modelNB.predict(x)
	 return ''.join(y)

		
def procesInputTex(inputText):
		inputDoc = inputText.lower()
		ps = PorterStemmer()
		inputDoc_class =[]
		matchingTokens =[]
		
		feature_value2 = features.to_dict()
		feature_value3 = {}
		document_class = []
				
		stemmed_feature_value2 = []
		stemmed_feature_value3 = []
		
		for t in feature_value2.keys():
			stemmed_feature_value2.append(ps.stem(t))
		
		for i, v in feature_value2.items():
			#print(i,v)
			feature_value3[ps.stem(i)]= v			
			
		tokens = nltk.word_tokenize(inputDoc)

		for token in tokens:
			if ps.stem(token) in stemmed_feature_value2:
			 
			 feature_value3[ps.stem(token)] = 1
			 matchingTokens.append(ps.stem(token))
			 			
			
		inputDoc_class.append(feature_value3)
					
		inputDataTransformed = pd.DataFrame(inputDoc_class)
						
		emptyfeatures = inputDataTransformed.drop(matchingTokens, axis = 1)
		
		cols = emptyfeatures.columns
				
					
		inputDataTransformed[cols] = inputDataTransformed[cols].apply(lambda x: 0).to_frame().T
		
			
		inputDataTransformed.to_excel(os.path.join(outputDir, 'UnseentestData.xlsx'))
		return inputDataTransformed
		
	
	
if __name__ == "__main__":
		#buildModel(creatFeatureVector())
		txt = input("Please enter your text: ")
		print('The search item likely to belong to category: ', predictCategory(procesInputTex(txt)))
		
		
	
	
