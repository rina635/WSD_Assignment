#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:08:23 2021

@author: Rina
"""
import nltk, string, re, math, sys
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import LidstoneProbDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import pandas as pd

# command line arguments for the file sources of training data, testing data, decision list
training_data = sys.argv[1]
testing_data = sys.argv[2]
my_decision_list = sys.argv[3]

# Ambiguous word
ambg_word = "line"

# initializing the decision list to an empty list
decision_list = []

# Function to preprocess the text (lower case, remove stopword, stemming and remove html)
def process_text(unprocessed_text):
    # lower case all unprocessed text
    unprocessed_text = unprocessed_text.lower()

    # save stopwords into a variable for removal
    sw_p = stopwords.words("english")
    
    # save punctuation symbols into the variable for removal 
    sw_p.extend(string.punctuation)
    
    # stem and replace other forms of the root word in the text for consistency
    ps = PorterStemmer()   # create a PorterStemmer class
    other_forms = 'lines'   # indicate other forms of the root word   
    unprocessed_text = unprocessed_text.replace(other_forms, ps.stem(other_forms))   # convert lines to line using the stemmed root word
    
    # remove sumbols that due to html tags
    processed_text = [re.sub(r'[\.\,\?\!\'\"\-\_/]','',w) for w in unprocessed_text.split(" ")]
    
    # remove stopwords and punctuation 
    processed_text = [w for w in processed_text if w not in sw_p and w != '']
    
    return processed_text   # return the processed text to the method that called it


# Extract data from XML into a list
def extract_training_data(file):
    with open(file, 'r') as data:
        soup_data = BeautifulSoup(data, 'html.parser')
    extracted_data = []
    for instance in soup_data.find_all('instance'):
        sentence = dict()
        sentence['id'] = instance['id']
        sentence['sense'] = instance.answer['senseid']
        text = ""
        for s in instance.find_all('s'):
            text = text + " "+ s.get_text()
        sentence['text'] = process_text(text)
        extracted_data.append(sentence)
        
    return extracted_data

# Extract test data from XML into list
def extract_test_data(file):
    with open(file, 'r') as data:
        soup_data = BeautifulSoup(data, 'html.parser')
    extracted_data = []
    for instance in soup_data.find_all('instance'):
        sentence = dict()
        sentence['id'] = instance['id']
        text = ""
        for s in instance.find_all('s'):
            text = text + " "+ s.get_text()
        sentence['text'] = process_text(text)
        extracted_data.append(sentence)
        
    return extracted_data


#This function extracts the collocative word for the ambigious word given n 
#and writes our feature rule into the decision list
#Adapted from https://pythonexamples.org/python-find-index-of-item-in-list/
def coll_rules(n, context, sense): #searches a LIST.
    coll_word_index = context.index(ambg_word) + n    #uses ambigious word's index to find the index of the collocative words
    if len(context) > coll_word_index and coll_word_index >= 0:
        rule = (str(n) + "_words_", context[coll_word_index], sense) #the rule is n words followed by the collocative word and the respective sense
        decision_list.append(rule) #adds the rule into the decision list initialized at the beginning.
    else:
        return ""
#Based on input n (# of words away from ambigious word) iterates through all
#the context sentences and senses in the training dataset to grab the collocative words
#with the respective senses
def change_coll(n):
    for i in range(0, len(train_context)):
        coll_rules(n, train_context[i], train_senses[i])
        
        
#Extracts the training data as a list       
train_data = extract_training_data(training_data)    
#Convert the training data into a dataframe
train_df = pd.DataFrame(train_data)    
#Creates a smaller dataframe of just the context
train_context = train_df['text']
#Creates a smaller dataframe of just the senses
train_senses = train_df['sense']
       
#Will execute for collocative words at -3 to 3 words away from the ambigious word
#and add it to the decisions list
#adapted from https://stackoverflow.com/questions/24089924/skip-over-a-value-in-the-range-function-in-python
for i in range (-3, 3):
    if i == 0:
        i = i + 1
        continue
    change_coll(i)

#Calculates the rule frequency
#can delete if you want, not important really.
rule_frequency = nltk.FreqDist(decision_list)

#want to know if the condition - collocative word affects the outcome - sense id


#https://stackoverflow.com/questions/15145172/nltk-conditionalfreqdist-to-pandas-dataframe
#Convert the decision list into a dataframe
dec_df = pd.DataFrame(decision_list) 
#Renames the columns for readability
dec_df.columns = ['w', 'coll', 'sense']
#Want the condition to be the collocative location and the word so combining them into one column
dec_df["condition"] = dec_df["w"] + dec_df["coll"]
#The new dataframe will have just the 2 columns.
decision_df = dec_df[['condition', 'sense']]

#Converting the dataframe back to a list so that there's no seperation between n and the coll_word
dec_list = decision_df.values.tolist()
#Performing the conditional frequency for the decision list of rules.
#https://lost-contact.mit.edu/afs/cs.pitt.edu/projects/nltk/docs/tutorial/probability/conditionalfreqdist.html
dec_cfd = nltk.ConditionalFreqDist(dec_list) 
#use the conditional frequency variable alongise the ELEprobability to compute the cpd
cpd = ConditionalProbDist(dec_cfd, LidstoneProbDist, 0.1)

#to see some of the conditions - Will delete later just for viewing.
for condition in dec_cfd:
    for word in dec_cfd[condition]:
        print (word, dec_cfd[condition].freq(word), condition)
        #printing word - which is sense and its frequency, and then the condition.
#https://stackoverflow.com/questions/62603854/conditional-frequency-distribution-using-browns-corpus-nltk-python
table_cfd = dec_cfd.tabulate(condition, sense = ['phone','product'])
#first column is the condition, coll_word, and # time its used for sense1 (phone), 
# and then 3rd column is times its used for sense2 (product)
#Don't really need this table just did it to check it out.


