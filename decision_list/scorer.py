'''
AIT 590 - Assignment 3 
Team 3 - Rafeef Baamer, Ashish Hingle, Rina Lidder, & Andy Nguyen
Date: 3/31/2021

Description: This Python program scores the decision list created by decision_list.py. A confusion matric is generated based on the results of the 
trained and tested data to be compared against a baseline.

Libraries used: RE, SYS, PANDAS

Usage Instructions: This program must be used in combination with the scorer.py program. To use the program:
1) Ensure all supplmentary files are placed in the correct directory
2) In the terminal, run the command "python scorer.pl my-line-answers.txt line-answers.txt"
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
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# command line arguments for the file sources of results obtained from decision-list and available gold standard
generated_file = sys.argv[1]
baseline_file = sys.argv[2]
testing_data = sys.argv[3]

#This function is to search each line from the outputs to create a dictionary of
#the sysnets and word sense. NEW FUNCTION**
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

#This function calculates the number of correct matches between the WSD model's output
#and the predicted answers. It then uses the number of correct matches to calculate the accuracy.
def accuracy_sense():
    correct = 0
    for key in keys:
        if(current_sense[key]) == pred_sense[key]:
            correct = correct + 1
    accuracy = (correct/len(keys))*100
    accuracy = round(accuracy,2)
    return accuracy
    
# Copy of extracting test data function to determine most ambiguous context
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


# open the two files taken as input from command line and strip out '\n'
with open(baseline_file, 'r') as data:
    mylist1 = [line.rstrip('\n') for line in data]
current_sense, keys = all_word_senses(mylist1)

with open(generated_file, 'r') as data:
    mylist2 = [line.rstrip('\n') for line in data]
pred_sense, keys = all_word_senses(mylist2)

total = len(keys)
baseline_count = 0
for key in keys:
    if(current_sense[key] == 'phone'):
        baseline_count = baseline_count + 1
baseline_acc = round((baseline_count/total)*100,2)
print("Baseline Accuracy: "+str(baseline_acc)+"%") 

accuracy_output = print('Overall Accuracy: {}%'.format(accuracy_sense()))
#creating array for our output and append the values to list


#Access the dictionary to retreive the predicted senses to a list.    

#creating array for gold standard and append the values to its list'''


#access the dicitionary values for the current senses into a list 
#https://realpython.com/python-dicts/#dvalues   
pred_list = list(pred_sense.values())
answer_list = list(current_sense.values())

# creating dataframes for both the files
df1 = pd.Series( (v for v in pred_list) )
df2 = pd.Series( (v for v in answer_list) )

#generating confusion matrix
df_confusion = pd.crosstab(df1, df2)
print("Confusion matrix is\n"  +str(df_confusion))

# Extract testing data into list
test_data = extract_test_data(testing_data)
#print(test_data)

#for element in test_data:
#    id1 = element['id']
#    text = element['text']

incorrect_senses = []   
for key in keys:
    if(current_sense[key]) != pred_sense[key]:
        print(len(current_sense.items()))
        #key, value = current_sense.items()
        #print(key)
        #incorrect_senses.append(value)
        
        #print(current_sense.keys())
        #for item in test_data:
        #    print(item.get('text'))
            #print(list(current_sense.keys())[0])
        #print(test_data.get(current_sense.keys()))
incorrect_senses = incorrect_senses[0]
print(incorrect_senses)
#for key in keys:
#    if(current_sense[key]) != pred_sense[key]:
#        print(current_sense[key])