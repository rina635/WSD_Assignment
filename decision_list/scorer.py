'''
AIT 590 - Assignment 3 

Team 3 - Rafeef Baamer, Ashish Hingle, Rina Lidder, & Andy Nguyen

This is the Scorer


'''





import pandas as pd
import re
import sys


# command line arguments for the file sources of results obtained from decision-list and available gold standard
my_key = sys.argv[1]
gs_key = sys.argv[2]

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

'''def get_senses(mylist):
    sense = {}
    keys = []
    for string in mylist:
        search = re.search('<answer instance="(.*)" senseid="(.*)"/>', string, re.IGNORECASE)
        key = search.group(1)
        keys.append(key)
        value = search.group(2)
        sense[key] = value
    return sense, keys'''
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

#open the two files taken as input from command line and strip out '\n'
with open(gs_key, 'r') as data:
    mylist1 = [line.rstrip('\n') for line in data]
current_sense, keys = all_word_senses(mylist1)

with open(my_key, 'r') as data:
    mylist2 = [line.rstrip('\n') for line in data]
pred_sense, keys = all_word_senses(mylist2)



'''
# For a specific key check if the values matches
correct = 0
total = len(keys)
for key in keys:
    if(answers[key] == preds[key]):
        correct += 1
correct

#caluclated the most frequent sense baseline
baseline_count = 0
for key in keys:
    if(answers[key] == 'phone'):
        baseline_count += 1
baseline_acc = (float(baseline_count)/float(total))*100
print("Baseline accuracy is "+str(baseline_acc)+"%") - Do we need baseline?


# calucalate the accuracy after learning features
accuracy = (float(correct)/float(total))*100
print("Accuracy after adding learned features is "+str(accuracy)+"%")
'''
accuracy_output = print('Based on the collocative features extracted the accuracy is {}%'.format(accuracy_sense()))
#creating array for our output and append the values to list
'''pred_list = []
for v in pred_sense:
    pred_list.append(pred_sense[v])'''
#Access the dictionary to retreive the predicted senses to a list.    
pred_list = list(pred_sense.values())
#creating array for gold standard and append the values to its list
'''answers_list = []
for v in current_sense:
    answers_list.append(current_sense[v])'''
#access the dicitionary values for the current senses into a list 
#https://realpython.com/python-dicts/#dvalues   
answer_list = list(current_sense.values())
# creating dataframes for both the files
df1 = pd.Series( (v for v in pred_list) )
df2 = pd.Series( (v for v in answer_list) )

#generating confusion matrix
df_confusion = pd.crosstab(df1, df2)
print("Confusion matrix is\n"  +str(df_confusion))
