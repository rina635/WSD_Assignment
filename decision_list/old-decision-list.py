'''
AIT 590 - Assignment 3 
Team 3 - Rafeef Baamer, Ashish Hingle, Rina Lidder, & Andy Nguyen
Date: 3/31/2021

Description: This Python program implements a decision list classifier to perform word sense disambiguation. The methodology used is based on Yarowsky's method for WSD
(Paper titled "DECISION LISTS FOR LEXICAL AMBIGUITY RESOLUTION: Application to Accent Restoration in Spanish and French). The program uses 6 features from a training set
of data ranging from -3 to +3, based on the paper. The program uses a training file with answers to instruct the sense of an ambiguous word.

Libraries used: NLTK, RE, MATH, SYS, BS4

Usage Instructions: This program must be used in combination with the scorer.py program. To use the program:
1) Ensure all supplmentary files are placed in the correct directory
2) In the terminal, run the command "python decision-list.py line-train.xml line-test.xml my-decision-list.txt > my-line-answers.txt"
	a) decision-list.py -> this file
	b) line-train.xml -> file that contains training data with answers for each sense of the ambiguous word
	c) line-test.xml -> file that contains test data (no answers present)
	d) my-decision-list.xml -> a log file used during debugging
	e) my-line-answers.txt -> file containing generated answers and sense for test data based on the training data
 
Algorithm defined in program:
Step 1: Extract the training data from XML files (preprocesses stopwords, punctuation, case, and symbols during extraction).
Step 2: Store conditional probability distributions of training data in decision list.
Step 3: Search the training data to count the frequency of each sense.
Step 4: Calculate the majority sense based frequency likelihood for rules
Step 5: Extract the testing data from XML files (preprocesses stopwords, punctuation, case, and symbols during extraction).
Step 6: Use majority sense to predict the test data
Step 7: Store decision list into my-line-answers.txt file

Additional features:  


Results of confusion Matrix:
Baseline accuracy: 57.14%
Based on the collocative features extracted the accuracy is 74.6%
Confusion matrix is
col_0    phone  product
row_0
phone       51       11
product     21       43

First 10 results of my-decision-list:
['Feature: -1 Word: telephone', 'Log-likelihood: 8.46', 'Sense: phone']
['Feature: -1 Word: access', 'Log-likelihood: 7.24', 'Sense: phone']
['Feature: -1 Word: car', 'Log-likelihood: -6.51', 'Sense: product']
['Feature: -1 Word: end', 'Log-likelihood: 6.34', 'Sense: phone']
['Feature: -1 Word: computer', 'Log-likelihood: -5.93', 'Sense: product']
['Feature: -1 Word: came', 'Log-likelihood: 5.93', 'Sense: phone']
['Feature: -1 Word: ps2', 'Log-likelihood: -5.93', 'Sense: product']
['Feature: 1 Word: dead', 'Log-likelihood: 5.93', 'Sense: phone']
['Feature: -2 Word: telephone', 'Log-likelihood: 5.67', 'Sense: phone']
['Feature: -1 Word: gab', 'Log-likelihood: 5.67', 'Sense: phone']

First 10 results my-line-answers.txt:
<answer instance="line-n.w8_059:8174:" senseid="phone"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>
<answer instance="line-n.w9_40:10187:" senseid="product"/>
<answer instance="line-n.w9_16:217:" senseid="product"/>
<answer instance="line-n.w8_119:16927:" senseid="product"/>
<answer instance="line-n.w8_008:13756:" senseid="product"/>
<answer instance="line-n.w8_041:15186:" senseid="phone"/>
<answer instance="line-n.art7} aphb 05601797:" senseid="phone"/>
<answer instance="line-n.w8_119:2964:" senseid="product"/>
    
Resources used for this assignment come from the materials provided in the AIT 590 course materials.
- Lecture powerpoints (AIT 590)
- Stanford University Prof. Dan Jurafsky's Video Lectures (https://www.youtube.com/watch?v=zQ6gzQ5YZ8o)
- Joe James Python: NLTK video series (https://www.youtube.com/watch?v=RYgqWufzbA8)
- w3schools Python Reference (https://www.w3schools.com/python/)
- regular expressions 101 (https://regex101.com/)
- dictionary search (https://www.kite.com/python/answers/how-to-search-if-dictionary-value-contains-certain-string-in-python)

'''

#Import libraries
import nltk, string, re, math, sys
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import LidstoneProbDist
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
import time

# command line arguments for the file sources of training data, testing data, decision list
training_data = sys.argv[1]
testing_data = sys.argv[2]
my_decision_list = sys.argv[3]

#stores the current time of the request from user as the 'start time'
start_time = time.time()

# Ambiguous word
ambg_word = "line"

# initializing the decision list to an empty list
decision_list = []

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

# this function finds the index of the collocative word
def find_coll(n, context):
    n_word_index = context.index(ambg_word) + n    
    if len(context) > n_word_index and n_word_index >= 0:
        return context[n_word_index]
    else:
        return ""

# This function adds the new conditions based on collocation to the decision list
def write_cond(cfd, data, n):
    for element in data:
        sense, context = element['sense'], element['text']
        coll_word = find_coll(n, context)
        if coll_word != '':
            condition = str(n) + " Word: " + coll_word
            cfd[condition][sense] = cfd[condition][sense] + 1
    return cfd

# To calculate the logarithm of likelihood for ratio of sense probabilities
def log_likelihood(cpd, rule):
    s1_prob = cpd[rule].prob("phone")
    s2_prob = cpd[rule].prob("product")
    if (s1_prob/s2_prob) != 0:
        return round(math.log((s1_prob/s2_prob),2),2)
    else:
        return 0

# checking whether the rule is satisfied in a given context
def check_rule(context, rule):
    rule_scope, rule_type, rule_feature = rule.split(" ")
    print("scope:")
    print(rule_scope)
    rule_scope = int(rule_scope)
    print("2nd: ")
    print(rule_scope)
    
    return find_coll(rule_scope, context) == rule_feature
        
# Function to predict the sense on test data
def predict(context, majority_label):
    for rule in decision_list:
        print(rule[0])
        if check_rule(context, rule[0]):
            if rule[1] > 0:
                return ("phone", context, rule[0])
            elif rule[1] < 0:
                return ("product", context, rule[0])
    return (majority_label, context, "default")

# extract the training fata from the XML file   
train_data = extract_training_data(training_data)

# Use conditional frequency distribution to add learned rules to the decision list
cfd = ConditionalFreqDist()
for i in range (-3, 3):
    if i == 0:
        i+=1
        continue
    cfd = write_cond(cfd, train_data, i)
        
# Instantiating Condition probability distribution to calculate the probabilities of the frequencies recorded above
cpd = ConditionalProbDist(cfd, LidstoneProbDist, 0.1)

# storing the learned rules into the decision list
for rule in cpd.conditions():

    likelihood = log_likelihood(cpd, rule)
    decision_list.append([rule, likelihood, "phone" if likelihood > 0 else "product"])
    
    decision_list.sort(key=lambda rule: math.fabs(rule[1]), reverse=True)

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
    else:
        print('The word does not exist.')

#retreives percentage frequency of each sense.
sensePercentage1 = round((sense1/textLen)*100,2)
sensePercentage2 = round((sense2/textLen)*100,2)

# Calculating the majority sense
majority_sense = "phone" if sense1 > sense2 else "product"

# extract the test fata from the XML file
test_data = extract_test_data(testing_data)

# Performing the predictions
for element in test_data:
    pred, _, r = predict(element['text'], majority_sense)
    id1 = element['id']
    print(f'<id="{id1}" sense="{pred}"/>')

# Storing the decision list into a file
writer = open(my_decision_list, 'w')   # open the text file

# loop through the decision list and write it to file with correct formatting
for i in decision_list:         
    i[0] = 'Feature: ' + i[0]                   # add 'Feature' text before printing feature 
    i[1] = 'Log-likelihood: ' + str(i[1])       # add 'Log-likelihood' before printing float
    i[2] = 'Sense: ' + i[2]                     # add 'Sense' before printing predicted sense
    writer.write('%s\n' % i)

writer.close()      # close the text file