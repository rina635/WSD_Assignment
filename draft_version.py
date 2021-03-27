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
