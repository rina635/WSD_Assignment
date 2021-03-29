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



#open the two files taken as input from command line and strip out '\n'
with open(gs_key, 'r') as data:
    mylist1 = [line.rstrip('\n') for line in data]
answers, keys = all_word_senses(mylist1)

with open(my_key, 'r') as data:
    mylist2 = [line.rstrip('\n') for line in data]
preds, keys = all_word_senses(mylist2)


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
print("Baseline accuracy is "+str(baseline_acc)+"%")


# calucalate the accuracy after learning features
accuracy = (float(correct)/float(total))*100
print("Accuracy after adding learned features is "+str(accuracy)+"%")


#creating array for our output and append the values to list
pred_list = []
for v in preds:
    pred_list.append(preds[v])
    

#creating array for gold standard and append the values to its list
answers_list = []
for v in answers:
    answers_list.append(answers[v])

# creating dataframes for both the files
df1 = pd.Series( (v for v in pred_list) )
df2 = pd.Series( (v for v in answers_list) )

#generating confusion matrix
df_confusion = pd.crosstab(df1, df2)
print("Confusion matrix is\n"  +str(df_confusion))
