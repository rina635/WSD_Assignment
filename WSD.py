#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:48:17 2021

@author: Rina
"""
import bs4 as bs  # BeautifulSoup
from bs4 import BeautifulSoup
import urllib.request
import re
import nltk
from nltk.tokenize import PunktSentenceTokenizer,sent_tokenize, word_tokenize
from nltk.corpus import stopwords

file = open("/Users/Rina/Documents/DAEMS/AIT590_NLP/WSD/line-data/line-train.xml", "r")
contents = file.read()

#Preprocessing function
def textclean(sentence):
    
    filtered_sent = []
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(sentence)
    
    for w in words:
        if w not in stop_words:
            filtered_sent.append(w)
            
    filtered_sent = [re.sub(r'[\.\,\?\!\'\"\-\_/$<>%]','',w) for w in filtered_sent]
    
    while ('' in filtered_sent): 
           filtered_sent.remove('')
    
    return filtered_sent


# parsing
soup = BeautifulSoup(contents, 'xml')
amg_word = soup.find_all('head')
sent = soup.find_all('s')
#Convert sentence to string
'''
sent = str(sent)
#preprocess sentences
clean_sent = textclean(sent) #we removed <> but it converted tags like <head> to word head

#POS tags each sentence token
tagged_s = nltk.pos_tag(new_sent2)
#new list contains all pos tags for the word line
l2 = [item for item in tagged_s if item[0].startswith('line')]
#https://stackoverflow.com/questions/7100243/finding-in-elements-in-a-tuple-and-filtering-them
#line is used as a NN or NNS
from collections import Counter
#Counts how many times line is nn or nns
#https://www.guru99.com/counting-pos-tags-nltk.html
counts = Counter( tag for word,  tag in l2)
#our pattern in sentences is that we have line and lines
all_counts = Counter( tag for word,  tag in tagged_s)


count_phone = Counter( tag for word,  tag in l2)

sent1 = soup.find_all('s')
sent1 = str(sent1)

#https://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

new_sent = cleanhtml(sent1) #removes html tags
new_sent2 = textclean(new_sent) #removes other issues
clean_new_sent = re.sub(r'[^\w\s\.]', '', new_sent)
Value_sent = sent_tokenize(clean_new_sent)

#METHOD - https://towardsdatascience.com/collocations-in-nlp-using-nltk-library-2541002998db
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
line_filter = lambda *w: 'line' not in w
finder = BigramCollocationFinder.from_words(new_sent2)
# only bigrams that appear 3+ times
finder.apply_freq_filter(3)
# only bigrams that contain 'line'
finder.apply_ngram_filter(line_filter)
# return the 10 n-grams with the highest PMI
# print (finder.nbest(bigram_measures.likelihood_ratio, 10))
for i in finder.score_ngrams(bigram_measures.likelihood_ratio):
    print (i)
    
## Trigrams
trigram_measures = nltk.collocations.TrigramAssocMeasures()
# Ngrams with 'creature' as a member
creature_filter = lambda *w: 'line' not in w
finder = TrigramCollocationFinder.from_words(
   new_sent2)
# only trigrams that appear 3+ times
finder.apply_freq_filter(3)
# only trigrams that contain 'creature'
finder.apply_ngram_filter(line_filter)
# return the 10 n-grams with the highest PMI
# print (finder.nbest(trigram_measures.likelihood_ratio, 10))
for i in finder.score_ngrams(trigram_measures.likelihood_ratio):
    print (i)    
'''
#parse out instances
inst = soup.find_all("instance")
#Creating list of all id numbers
id = []
for meta in inst:
    x = (meta.attrs['id'])
    id.append(x) #id has : at the end need to remove that.
print(id)
#Extracting specific senseids
answer = soup.find_all("answer")
senseid = []
for meta in answer:
    x = (meta.attrs['senseid'])
    senseid.append(x) 
    
#used to print parsings
'''for data in amg_word:
    print(data.get_text())

for data in sent:
    print(data.get_text()) '''   

#Create a dictionary of instance IDs and tokens as the values
keys_list = senseid
values_list = Value_sent
zip_iterator = zip(keys_list, values_list)

a_dictionary = dict(zip_iterator)
print(a_dictionary)
