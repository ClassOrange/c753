# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 23:45:03 2021

@author: Cindy
"""

from feature_format import featureFormat, target_feature_split
import pickle
from tester import dump_classifier_and_data
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.metrics import accuracy_score
from tester import test_classifier
from sklearn.model_selection import KFold

#%% Initial data load

with open('final_project_dataset.pkl', 'rb') as data_file:
    data_dict = pickle.load(data_file)
    
#%% create pandas data frame; .T is to transpose and use names as
# rows/features as columns

pdEmails = pd.DataFrame(data_dict).T

# remove the 'TOTAL' row as we know it's not valid
pdEmails = pdEmails.drop('TOTAL', axis = 0)

# need to be able to work with numerical data properly; int won't do NaN
cols = pdEmails.columns
noChange = ['poi', 'email_address']

for col in cols:
    if col not in noChange:
        pdEmails[col] = pdEmails[col].astype('float64')
        
#%% new feature creation

pdEmails['bon_sal'] = pdEmails['bonus'] / (pdEmails['bonus'] + pdEmails['salary'])
pdEmails['pct_email_poi'] = ((pdEmails['from_poi_to_this_person'] +
                              pdEmails['from_this_person_to_poi']) /
                             (pdEmails['from_messages'] + pdEmails['to_messages']))
pdEmails['pct_shared_poi'] = (pdEmails['shared_receipt_with_poi'] /
                              pdEmails['to_messages'])
        
#%% potential features for poi prediction

pdEmailsCut = pdEmails[['poi', 'salary', 'bonus', 'total_stock_value',
                     'bon_sal', 'pct_email_poi', 'pct_shared_poi']]

pdChosenNoNA = pdEmailsCut.dropna()
        
#%% final chosen set for poi prediction; reverting to data_dict

pdTestPSB = pdEmails[(pdEmails['total_stock_value'] < 2000000
    )][['poi', 'bonus','total_stock_value', 'pct_shared_poi']].dropna()
features_list = ['poi', 'bonus', 'total_stock_value', 'pct_shared_poi']
data_dict = pdTestPSB.T.to_dict()

#%% store to dataset for easy export below.

dataset = data_dict

# Extract features and labels from dataset for local testing
data = featureFormat(dataset, features_list, sort_keys=True)
labels, features = target_feature_split(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=12)
     
#%% Decision Tree classification (helped me choose features)

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print('accuracy: %f' %score )
print(clf.feature_importances_)

#%% NB Gaussian classification

def classify(features_train, labels_train, features_test, labels_test):
   
    clf = GaussianNB()
    t0 = time()
    clf.fit(features_train, labels_train)
    print("Training Time:", round(time()-t0, 3), "s")
    
    t0 = time()
    pred = clf.predict(features_test)
    print("Predicting Time:", round(time()-t0, 3), "s")
    
    accuracy = accuracy_score(pred, labels_test)
    return accuracy

clf = classify(features_train, labels_train, features_test, labels_test)
clf
#%% initiate pkl file creation/export

dump_classifier_and_data(clf, dataset, features_list)

#%% test my stuff

clf = DecisionTreeClassifier(min_samples_split = 0.05,
                             min_weight_fraction_leaf = 0.045,
                             min_impurity_decrease = 0.01)

test_classifier(clf, dataset, features_list)

        
    
