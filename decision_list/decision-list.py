#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIT 590 - Assignment 3 
Team 3 - Rafeef Baamer, Ashish Hingle, Rina Lidder, & Andy Nguyen
Date: 3/31/2021

Description: This Python program implements a decision list classifier to perform word sense disambiguation. The methodology used is based on Yarowsky's method for WSD
(Paper titled "DECISION LISTS FOR LEXICAL AMBIGUITY RESOLUTION: Application to Accent Restoration in Spanish and French). The program uses 6 features from a training set
of data ranging from -3 to +3, based on the paper. The program uses a training file with answers to instruct the sense of an ambiguous word.

Libraries used: NLTK, RE, MATH, SYS, BEAUTIFULSOUP, STRING, PANDAS

Additional features (for extra credit):
- Timer calculates time it takes for scorer.py to run (printed after running scorer.py)
- List of incorrect senses and the context (printed after running scorer.py)

Usage Instructions: This program must be used in combination with the scorer.py program. To use the program:
1) Ensure all supplmentary files are placed in the correct directory
2) In the terminal, run the command "python decision-list.py line-train.xml line-test.xml my-decision-list.txt > my-line-answers.txt"
	a) decision-list.py -> this file
	b) line-train.xml -> file that contains training data with answers for each sense of the ambiguous word
	c) line-test.xml -> file that contains test data (no answers present)
	d) my-decision-list.xml -> a log file used during debugging
	e) my-line-answers.txt -> file containing generated answers and sense for test data based on the training data
3) The scorer.py file is used in combination with this file to generate an confusion matrix and compare the results
 
Algorithm defined in program:
Step 1: Extract the training data from XML files (preprocesses stopwords, punctuation, case, and symbols during extraction).
Step 2: Store conditional probability distributions of training data in decision list.
Step 3: Search the training data to count the frequency of each sense.
Step 4: Calculate the majority sense based frequency likelihood for rules
Step 5: Extract the testing data from XML files (preprocesses stopwords, punctuation, case, and symbols during extraction).
Step 6: Use majority sense to predict the test data
Step 7: Store decision list into my-line-answers.txt file

Results of confusion Matrix:
Baseline Accuracy: 57.14%
Overall Accuracy: 74.6%
Confusion matrix is
col_0    phone  product
row_0
phone       51       11
product     21       43

First 10 results of my-decision-list:
{'condition': '-1_words_telephone', 'log_likelihood': 4.2626798770413155, 'sense': 'phone'}
{'condition': '-1_words_access', 'log_likelihood': 3.4339872044851463, 'sense': 'phone'}
{'condition': '-1_words_car', 'log_likelihood': -2.9444389791664403, 'sense': 'product'}
{'condition': '-1_words_end', 'log_likelihood': 2.833213344056216, 'sense': 'phone'}
{'condition': '-1_words_computer', 'log_likelihood': -2.5649493574615367, 'sense': 'product'}
{'condition': '-1_words_came', 'log_likelihood': 2.5649493574615367, 'sense': 'phone'}
{'condition': '-1_words_ps2', 'log_likelihood': -2.5649493574615367, 'sense': 'product'}
{'condition': '1_words_dead', 'log_likelihood': 2.5649493574615367, 'sense': 'phone'}
{'condition': '-2_words_telephone', 'log_likelihood': 2.3978952727983707, 'sense': 'phone'}
{'condition': '-1_words_gab', 'log_likelihood': 2.3978952727983707, 'sense': 'phone'}

First 10 results my-line-answers.txt:
<id="line-n.w8_059:8174:" sense="phone"/>
<id="line-n.w7_098:12684:" sense="phone"/>
<id="line-n.w8_106:13309:" sense="phone"/>
<id="line-n.w9_40:10187:" sense="product"/>
<id="line-n.w9_16:217:" sense="product"/>
<id="line-n.w8_119:16927:" sense="product"/>
<id="line-n.w8_008:13756:" sense="product"/>
<id="line-n.w8_041:15186:" sense="phone"/>
<id="line-n.art7} aphb 05601797:" sense="phone"/>
<id="line-n.w8_119:2964:" sense="product"/>

Resources used for this assignment come from the materials provided in the AIT 590 course materials.
- Lecture powerpoints (AIT 590)
- Stanford University Prof. Dan Jurafsky's Video Lectures (https://www.youtube.com/watch?v=zQ6gzQ5YZ8o)
- Joe James Python: NLTK video series (https://www.youtube.com/watch?v=RYgqWufzbA8)
- w3schools Python Reference (https://www.w3schools.com/python/)
- regular expressions 101 (https://regex101.com/)
- dictionary search (https://www.kite.com/python/answers/how-to-search-if-dictionary-value-contains-certain-string-in-python)
- sorting dictionary using lambda https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value-of-the-dictionary
"""

# import libraries
import nltk, string, re, math, sys
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, ELEProbDist
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
    
    # remove symbols that due to html tags
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
 
# Based on input n (# of words away from ambigious word) iterates through all the context sentences
# and senses in the training dataset to grab the collocative words with the respective senses
def change_coll(n):
    for i in range(0, len(train_context)):
        coll_rules(n, train_context[i], train_senses[i])
        
#Creates a list ccontaining a dictionary of the conditions, log likelihood, and
#respective senses.
def create_dict(n, j, s):
    new_dic = []
    for n, j, s in zip(cond_list, log_likelihood, sense_assign):
        x = { 'condition': n, 'log_likelihood': j, 'sense': s }
        new_dic.append(x)
    return new_dic
    
# Find the index for the given rule
def find_index(context, rule):
    delimiter = rule.split("_")
    rule_index = int(delimiter[0])
    
    return find_coll(rule_index, context) == delimiter[2]

# Finds the word at the given index
def find_coll(coll, context):
    coll_word_index = context.index(ambg_word) + coll    
    if len(context) > coll_word_index and coll_word_index >= 0:
        return context[coll_word_index]
    else:
        return ""
        
# Function to determine sense on given context
def determine_sense(most_likely_sense, context):
    for rule in dec_list:
        if find_index(context, list(rule.values())[0]):
            if rule.get(list(rule)[1]) > 0:
                return ("phone", context, rule.get(list(rule)[0]))
            elif rule.get(list(rule)[1]) < 0:
                return ("product", context, rule.get(list(rule)[0]))
    return (most_likely_sense, context, "default")
        
# Step 1: Extract the training data from XML files (preprocesses stopwords, punctuation, case, and symbols during extraction)       
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

# Step 2: Store conditional probability distributions of training data in decision list.
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
# Creating dataframe of conditional frequency
#https://stackoverflow.com/questions/62603854/conditional-frequency-distribution-using-browns-corpus-nltk-python
df_cfdist = pd.DataFrame.from_dict(dec_cfd , orient='index')
#https://www.kite.com/python/docs/nltk.ConditionalProbDist
#https://lost-contact.mit.edu/afs/cs.pitt.edu/projects/nltk/docs/ref/nltk.probability.ELEProbDist.html
cpdist = ConditionalProbDist(dec_cfd , ELEProbDist, 2)

# Create a list of the senses for the ambiguous word
# Grab cpdist values of the senses and the conditions and storing it in a separate list
cond_list = []
sense1_list  = []
sense2_list = []
for item in cpdist.conditions():
    cond = item
    phone_prob = cpdist[item].prob("phone")
    prod_prob = cpdist[item].prob("product")
    sense1_list.append(phone_prob)
    sense2_list.append(prod_prob)
    cond_list.append(cond)

# Calculating log likelihood
# Loops through the conditional probabilities of each sense and divides them
div_probs = [i / j for i, j in zip(sense1_list, sense2_list)]    
log_likelihood = []
# Takes the log and absolute value of previous values to calculate the log likelihood
for i in div_probs:
    if i == 0:
        log_likelihood.append(0)
    else:
        x = math.fabs(i)
        x = math.log(x)
        x = round(x, 2)
        log_likelihood.append(x)

# Assigning sense to each condition
sense_assign = []
for item in log_likelihood:
    if item >= 0:
        sense = "phone"
        sense_assign.append(sense)
    else:
        sense = "product"
        sense_assign.append(sense)

# Step 3: Search the training data to count the frequency of each sense.    
# extract the training data from the XML file   
train_data = extract_training_data(training_data)

#Searches the training data to count the frequency of each sense.
#Method adapted from #https://www.kite.com/python/answers/how-to-search-if-dictionary-value-contains-certain-string-in-python
sense1 = 0
sense2 = 0
textLen = len(train_data)
for i in range(0, len(train_data)):
    if 'phone' in list(train_data[i].values()):
        sense1 = sense1 + 1
    elif 'product' in list(train_data[i].values()):
        sense2 = sense2 + 1

# Step 4: Calculate the majority sense based frequency likelihood for rules
# Calculating which sense occurs most often
most_likely_sense = "phone" if sense1 > sense2 else "product"

# Creating the decision dictionary with the condition, log likelihood, and sense
dec_list = create_dict(cond_list, log_likelihood, sense_assign)

# Sorting decision dictionary in reverse
sorted_dec_list = sorted(dec_list, key=lambda k: math.fabs(k['log_likelihood']), reverse=True)

# Step 5: Extract the testing data from XML files (preprocesses stopwords, punctuation, case, and symbols during extraction).
# Extract testing data into list
test_data = extract_test_data(testing_data)

# Step 6: Use majority sense to predict the test data
# Performing prediction on testing data
for element in test_data:
    pred = determine_sense(most_likely_sense, element['text'])
    id1 = element['id']
    print(f'<answer instance="{id1}" senseid="{pred[0]}"/>')

# Step 7: Store decision list into my-line-answers.txt file
# Storing the decision list into a file
writer = open(my_decision_list, 'w')   # open the text file

# loop through the decision list and write it to file with correct formatting
for i in sorted_dec_list:         
    writer.write('%s\n' % i)

writer.close()