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

sent = str(sent)
#preprocess sentences
clean_sent = textclean(sent)
#parse out instances
inst = soup.find_all("instance")
#Creating list of all id numbers
id = []
for meta in inst:
    x = (meta.attrs['id'])
    id.append(x)
    
#used to print parsings
'''for data in amg_word:
    print(data.get_text())

for data in sent:
    print(data.get_text()) '''   

#Create a dictionary of instance IDs and tokens as the values
dict = {}
keys = id
values = clean_sent

d_test = dict(zip(keys, values))
