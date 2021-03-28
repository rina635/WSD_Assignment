'''
AIT 590 - Assignment 3 

Team 3 - Rafeef Baamer, Ashish Hingle, Rina Lidder, & Andy Nguyen




'''


#Import libraries
import nltk, string, re, math, sys
from nltk.probability import ConditionalFreqDist
from nltk.probability import ConditionalProbDist
from nltk.probability import LidstoneProbDist
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


# command line arguments for the file sources of training data, testing data, decision list
training_data = sys.argv[1]
testing_data = sys.argv[2]
my_decision_list = sys.argv[3]

# Ambiguous word
ambg_word = "line"

# initializing the decision list to an empty list
decision_list = []


# Function to preprocess the textual content
def process_text(text):
    text = text.lower()

    # removing the standard stop word from the text
    stop_words = stopwords.words("english")
    stop_words.extend(string.punctuation)
    
    # treating "lines" and "line" as a single entity
    text = text.replace("lines", "line")
    corpus = [re.sub(r'[\.\,\?\!\'\"\-\_/]','',w) for w in text.split(" ")]
    corpus = [w for w in corpus if w not in stop_words and w != '']
    return corpus


# This function is to retrieve the collocative words to the ambigious word.
def find_coll(n, context):
    ambg_index = context.index(ambg_word)
    n_word_index = ambg_index + n
    return context[n_word_index] #indexes the context to get the word


# This function adds the new conditions based on collocation to the decision list
def write_cond(cfd, data, n):
    for element in data:
        sense, context = element['sense'], element['text']
        n_word = find_coll(n, context)
        if n_word != '':
            cond = 'Position: {}w {}'.format(n, n_word)
            cfd[cond][sense] += 1
    return cfd


# To calculate the logarithm of likelihood for ratio of sense probabilities
def log_likelihood(cpd, rule):
    prob = cpd[rule].prob("phone")
    prob_star = cpd[rule].prob("product")
    div = prob / prob_star
    if div == 0:
        return 0
    else:
        return math.log(div, 2)


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


# Extracting the textual content from training data through XML parsing
with open(training_data, 'r') as data:
    soup = BeautifulSoup(data, 'html.parser')
train_data = []
for instance in soup.find_all('instance'):
    sntnc = dict()
    sntnc['id'] = instance['id']
    sntnc['sense'] = instance.answer['senseid']
    text = ""
    for s in instance.find_all('s'):
        text = text+ " "+ s.get_text()
    sntnc['text'] = process_text(text)
    train_data.append(sntnc)

# Use conditional frequency distribution to add learned rules to the decision list
cfd = ConditionalFreqDist()
cfd = write_cond(cfd, train_data, 1)
cfd = write_cond(cfd, train_data, -1)
cfd = write_cond(cfd, train_data, 2)
cfd = write_cond(cfd, train_data, -2)
cfd = write_cond(cfd, train_data, 3)
cfd = write_cond(cfd, train_data, -3)



# Instantiating Condition probability distribution to calculate the probabilities of the frequencies recorded above
cpd = ConditionalProbDist(cfd, LidstoneProbDist, 0.1)

# storing the learned rules into the decision list
for rule in cpd.conditions():

    likelihood = log_likelihood(cpd, rule)
    decision_list.append([rule, likelihood, "phone" if likelihood > 0 else "product"])
    
    decision_list.sort(key=lambda rule: math.fabs(rule[1]), reverse=True)


# extracting the test data through XML parsing
with open(testing_data, 'r') as data:
    test_soup = BeautifulSoup(data, 'html.parser')

test_data = []
for instance in test_soup('instance'):
    sntnc = dict()
    sntnc['id'] = instance['id']
    text = ""
    for s in instance.find_all('s'):
        text = text+ " "+ s.get_text()
    sntnc['text'] = process_text(text)
    test_data.append(sntnc)

# Calculating the frequencies of each senses
senseA, senseB = 0.0, 0.0
for element in train_data:
    if element['sense'] == "phone":
        senseA += 1.0
    elif element['sense'] == 'product':
        senseB += 1.0
    else:
        print("warning no match")

# Calculating the majority sense
majority_sense = "phone" if senseA > senseB else "product"

# Performing the predictions
predictions = []
for element in test_data:
    pred, _, r = predict(element['text'], majority_sense)
    id1 = element['id']
    predictions.append(f'<answer instance="{id1}" senseid="{pred}"/>')
    print(f'<answer instance="{id1}" senseid="{pred}"/>')


# Storing the decision list into a file
with open(my_decision_list, 'w') as output:  
    for listitem in decision_list:
        output.write('%s\n' % listitem)

