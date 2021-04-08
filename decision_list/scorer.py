'''
AIT 590 - Assignment 3 
Team 3 - Rafeef Baamer, Ashish Hingle, Rina Lidder, & Andy Nguyen
Date: 3/31/2021

Description: This Python program scores the decision list created by decision_list.py. A confusion matric is generated based on the results of the 
trained and tested data to be compared against a baseline.

Libraries used: RE, SYS, PANDAS

Usage Instructions: This program must be used in combination with the scorer.py program. To use the program:
1) Ensure all supplmentary files are placed in the correct directory
2) In the terminal, run the command "python scorer.py my-line-answers.txt line-answers.txt" or "python scorer.py my-line-answers.txt line-answers.txt line-text.xml"
	a) scorer.py -> this file
	b) my-line-answers.txt -> generated file from decision-list.py
	c) line-answers.txt -> baseline file with ansers for each sense
3) The baseline accuracy, accuracy after adding features and confusion matrix are printed.
 
Algorithm defined in program:
Step 1: Retrieve the my-line-answers.txt data file created by decision-list.py as predicted sense
Step 2: Retrieve the line-answers.txt data file as current sense
Step 3: Calculate and print the baseline accuracy based on the line-answers.txt file provided
Step 4: Create and print a confusion matrix based on the my-line-answers.txt file provided

Results of confusion Matrix:
Baseline Accuracy: 57.14%
Overall Accuracy: 74.6%
Confusion matrix is
col_0    phone  product
row_0
phone       51       11
product     21       43

First 10 results of my-decision-list:
<answer instance="line-n.w8_059:8174:" senseid="phone"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>
<answer instance="line-n.w9_40:10187:" senseid="phone"/>
<answer instance="line-n.w9_16:217:" senseid="phone"/>
<answer instance="line-n.w8_119:16927:" senseid="product"/>
<answer instance="line-n.w8_008:13756:" senseid="phone"/>
<answer instance="line-n.w8_041:15186:" senseid="phone"/>
<answer instance="line-n.art7} aphb 05601797:" senseid="phone"/>
<answer instance="line-n.w8_119:2964:" senseid="product"/>

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

'''
import pandas as pd
import importlib
import re
import sys
import string, time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

#The command line arguments to run the scorer using the results obtained from decision-list.py (generated_file)
#and the gold standard "key" data (key_file)
generated_file = sys.argv[1]
key_file = sys.argv[2]
if len(sys.argv) > 3:
    testing_data = sys.argv[3]
else:
    testing_data = "line-test.xml"

#stores the current time of the request from user as the 'start time'
start_time = time.time()

#This function is to search each line from a file and create a dictionary of
#the sysnets (key) and word sense (value) by spliting at the underscore character.
def all_word_senses(out_list):
    sense = {}
    keys = []
    for line in out_list:
        split_lines = re.split('"', line)
        key = split_lines[1]
        keys.append(key)
        value = split_lines[3]
        sense[key] = value
    return sense, keys

#This function calculates the accuracy of the model using the number of correct matches between the WSD model's 
#output (pred_senses) and the key data file (correct_sense)
def accuracy_sense():
    correct = 0
    for key in keys:
        if(correct_sense[key]) == pred_sense[key]:
            correct = correct + 1
    accuracy = (correct/len(keys))*100
    accuracy = round(accuracy,2)
    return accuracy

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

#Key value for gold standard key data is: Key and Correct_sense
with open(key_file, 'r') as data:
    mylist1 = [line.rstrip('\n') for line in data]
correct_sense, keys = all_word_senses(mylist1)

#Key value pair fore the generated file : Key and Pred_sense
with open(generated_file, 'r') as data:
    mylist2 = [line.rstrip('\n') for line in data]
pred_sense, keys = all_word_senses(mylist2)

#Uses the total number of keys to calculate the baseline accuracy
total = len(keys)
baseline_count = 0
for key in keys:
    if(correct_sense[key] == 'phone'):
        baseline_count = baseline_count + 1
baseline_acc = round((baseline_count/total)*100,2)
print("Baseline Accuracy: "+str(baseline_acc)+"%") 

#Prints the accuracy output in the command line.
accuracy_output = print('Overall Accuracy: {}%\n'.format(accuracy_sense()))

#access the dicitionary values for the predicted sense and correct senses into a list 
#https://realpython.com/python-dicts/#dvalues   
pred_list = list(pred_sense.values())
correct_list = list(correct_sense.values())

#Converts each of the lists into a dataframe for the confusion matrix.
generated_file_df = pd.Series((v for v in pred_list))
key_file_df = pd.Series((v for v in correct_list))

#generating confusion matrix
df_confusion = pd.crosstab(generated_file_df, key_file_df)
print("Confusion matrix:\n"  +str(df_confusion))

# Extract testing data into list
test_data = extract_test_data(testing_data)

# creating a list of incorrect senses based on predicitons
incorrect_senses = []   
for key in keys:
    if correct_sense[key] != pred_sense[key]:
        incorrect_senses.append(key)

# printing incorrect sense predictions with corresponding context
print('\nBelow listed are the incorrectly classified senses and their contents:')
incorrect_context = []
for context in test_data:
    id1 = context["id"]
    if id1 in incorrect_senses:
        print("ID: " + str(context["id"]))
        print("Text: " + str(context["text"]))

stop_time = time.time()
#subtract the two time periods to calculate the runtime of the scorer:
print('\nTime Executed :', (stop_time - start_time), 'seconds.')