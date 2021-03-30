'''
AIT 590 - Assignment 3 
Team 3 - Rafeef Baamer, Ashish Hingle, Rina Lidder, & Andy Nguyen
Date: 3/31/2021
Description:
Libraries used: 

Additional features: 

Usage Instructions:

Algorithm defined in program:


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
print(sys.argv[1])
print(sys.argv[2])
training_data = sys.argv[1]
testing_data = sys.argv[2]
my_decision_list = sys.argv[3]
#stores the current time of the request from user as the 'start time'
start_time = time.time()
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

#this function finds the index of the collocative word
def find_index(context, word):
    word_index = []
    i = 0
    while True:
        try: 
            i = context.index(word, i) + 1
            word_index.append(i) 
        except ValueError:
            return word_index
        

# This function is to retrieve the collocative words based on n - # words away from ambigious word
def find_coll(n, context):
    coll_word_index = find_index(context, ambg_word)
    coll_word_index = coll_word_index[0]
    if len(context) > coll_word_index and coll_word_index >=0:
        return context[coll_word_index] #indexes the context to get collocative word
    else:
        return "" #if n is out of bounds it will return a blank'  


# This function adds the new conditions based on collocation to the decision list
def write_cond(cfd, data, n):
    for element in data:
        sense, context = element['sense'], element['text']
        coll_word = find_coll(n, context)
        if coll_word != '':
            condition = str(n) + "_word_" + coll_word
            cfd[condition][sense] = cfd[condition][sense] + 1
    return cfd


# To calculate the logarithm of likelihood for ratio of sense probabilities
def log_likelihood(cpd, rule):
    s1_prob = cpd[rule].prob("phone")
    s2_prob = cpd[rule].prob("product")
    if (s1_prob/s2_prob) != 0:
        return math.log((s1_prob/s2_prob),2)
    else:
        return 0


# checking whether the rule is satisfied in a given context
def check_rule(context, rule):
    rule_scope, rule_type, rule_feature = rule.split("_")
    rule_scope = int(rule_scope)
    
    return find_coll(rule_scope, context) == rule_feature
        
# Function to predict the sense on test data
def predict(context, majority_label):
    for rule in decision_list:
        if check_rule(context, rule[0]):
            if rule[1] > 0:
                return ("phone", context, rule[0])
            elif rule[1] < 0:
                return ("product", context, rule[0])
    return (majority_label, context, "default")


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


# extracting the test data through XML parsing
#with open(testing_data, 'r') as data:
#    test_soup = BeautifulSoup(data, 'html.parser')

#test_data = []
#for instance in test_soup('instance'):
#    sntnc = dict()
#    sntnc['id'] = instance['id']
#    text = ""
#    for s in instance.find_all('s'):
#        text = text+ " "+ s.get_text()
#    sntnc['text'] = process_text(text)
#    test_data.append(sntnc)

test_data = extract_test_data(testing_data)

# Calculating the frequencies of each senses
#senseA, senseB = 0.0, 0.0
#for element in train_data:
 #   if element['sense'] == "phone":
  #      senseA += 1.0
   # elif element['sense'] == 'product':
    #    senseB += 1.0
    #else:
     #   print("warning no match")

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

#Stores time it took for model to process input as 'stop time'    
#stop_time = time.time()
#subtract the two time periods to calculate the runtime of the decision list
#prints the elapsed time in the my-line-answers file
#print('Time elapsed:', round((stop_time - start_time),3), 'sec')

#retreives percentage frequency of each sense.
sensePercentage1 = round((sense1/textLen)*100,2)
sensePercentage2 = round((sense2/textLen)*100,2)

#print(sensePercentage1, '% of instances of the word line is a phone')
#print(sensePercentage2, '% of instances of the word line is a product' + '\n')
      


# Calculating the majority sense
majority_sense = "phone" if sense1 > sense2 else "product"

# Performing the predictions
predictions = []
for element in test_data:
    pred, _, r = predict(element['text'], majority_sense)
    id1 = element['id']
    predictions.append(f'<answer instance="{id1}" senseid="{pred}"/>')
    print(f'<answer instance="{id1}" senseid="{pred}"/>')

# Storing the decision list into a file
writer = open(my_decision_list, 'w')   # open the text file

for i in decision_list:         # loop through the decision list and write it to file
    writer.write('%s\n' % i)

writer.close()      # close the text file

