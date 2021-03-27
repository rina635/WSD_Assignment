#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:52:37 2021

@author: Rina
"""
import bs4 as bs  # BeautifulSoup
from bs4 import BeautifulSoup
import urllib.request
import re
import nltk
from nltk.tokenize import PunktSentenceTokenizer,sent_tokenize, word_tokenize
from nltk.corpus import stopwords

#Removes html tags
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

#Preprocessing function - Tokenizes, removes punctuation and stop words:
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


file = open("/Users/Rina/Documents/DAEMS/AIT590_NLP/WSD/line-data/line-train.xml", "r")
contents = file.read()

# parsing
soup = BeautifulSoup(contents, 'xml')


#Giant list but keeps each instance of 'line' separate to maintain order of things
div_content = contents.split("</instance>")
#how to loop this through entire div_content? then we can do it to all 300+ instances.
answer1 = str(div_content[0]) #just the 1st collection - what needs to be split up.
sysnetid = re.findall(r"line-............", answer1) #gets the sysnet numberid
sense_id = re.findall(r"senseid=.*", answer1) 
#need to find a way to just get the word after senseid by itself
context = re.findall(r"<s.*", answer1) #captures the sentence of context.
context = cleanhtml(str(context)) #gets rid of html tags.

#once every part is separated out then put it into a dictionary.
keys_list = senseid
values_list = Value_sent
zip_iterator = zip(keys_list, values_list)

a_dictionary = dict(zip_iterator)
print(a_dictionary)


#need it to be sentence tokens for collocations:

sent = soup.find_all('s')
sent = str(sent)
new_sent = cleanhtml(sent)
new_sent2 = textclean(new_sent)

METHOD - https://towardsdatascience.com/collocations-in-nlp-using-nltk-library-2541002998db
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

