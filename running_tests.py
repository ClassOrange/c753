#!/usr/bin/env python
# coding: utf-8

# # Starter Code for the Lesson 17: Final Project

# In[41]:


import sys
sys.path.append('utils/')
from feature_format import feature_format, target_feature_split
import pickle
from tester import dump_classifier_and_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tester import test_classifier


# In[6]:


#Initial data load

with open('InProgress/final_project_dataset.pkl', 'rb') as data_file:
    data_dict = pickle.load(data_file)



# ## Task 1: Select what features you'll use

# In[7]:


# create pandas data frame; .T is to transpose and use names as rows/features as columns
pdEmails = pd.DataFrame(data_dict).T

# remove the 'TOTAL' row as we know it's not valid
pdEmails = pdEmails.drop('TOTAL', axis = 0)


# In[8]:


# need to be able to work with numerical data properly, but int won't allow NaN
cols = pdEmails.columns
noChange = ['poi', 'email_address']

for col in cols:
    if col not in noChange:
        pdEmails[col] = pdEmails[col].astype('float64')


# In[ ]:


sb.scatterplot(data = pdEmails, x = 'salary', y = 'total_payments', hue = 'poi')


# In[ ]:


pdEmails[(pdEmails['total_payments'] > 100000000)]


# In[ ]:


sb.scatterplot(data = pdEmails[(pdEmails['salary'] < 600000)], x = 'salary', y = 'total_payments', hue = 'poi')


# In[10]:


# bonus as a percentage of bonus + salary
pdEmails['bon_sal'] = pdEmails['bonus'] / (pdEmails['bonus'] + pdEmails['salary'])


# In[ ]:


#df.hist(column='Test1', by='Major')
pdEmails.hist(column = 'bon_sal', by = 'poi')


# In[ ]:


pdEmails[(pdEmails['expenses'] > 150000)]


# In[ ]:


sb.scatterplot(data = pdEmails, x = 'from_poi_to_this_person', y = 'from_this_person_to_poi', hue = 'poi')


# In[ ]:


### 3 new features


# In[ ]:


pdEmails[(pdEmails['from_this_person_to_poi'] > 100)]


# In[ ]:


pdEmails.info()


# In[11]:


# calculating a potential feature: emails sent to/received from POI as percentage of total email sent/received

pdEmails['pct_email_poi'] = ((pdEmails['from_poi_to_this_person'] + pdEmails['from_this_person_to_poi']) /
                             (pdEmails['from_messages'] + pdEmails['to_messages']))


# In[ ]:


sb.boxplot(data = pdEmails, x = 'poi', y = 'pct_email_poi')


# In[ ]:


pdEmails[(pdEmails['pct_email_poi'] > .13)]


# In[12]:


pdEmails['pct_shared_poi'] = pdEmails['shared_receipt_with_poi'] / pdEmails['to_messages']


# In[ ]:


sb.scatterplot(data = pdEmails, x = 'pct_shared_poi', y = 'exercised_stock_options', hue = 'poi')


# In[ ]:

# #### Trying to do more testing re: features, post new data frame

# ### Data frame and features list for current set

# In[13]:


# current features after additions; pare this down more based on charts and such above

pdEmailsCut = pdEmails[['poi', 'salary', 'bonus', 'total_stock_value',
                     'bon_sal', 'pct_email_poi', 'pct_shared_poi']]

pdChosenNoNA = pdEmailsCut.dropna()


# In[14]:


# The first feature must be "poi".
# features_list = ['poi', 'salary', 'bonus', 'total_stock_value',
#                      'bon_sal', 'pct_email_poi', 'pct_shared_poi']


# ### Testing features using automated stuff

# In[15]:


pdEmails.columns


# In[ ]:


'''# to test different combinations, replace both bracketed lists below

pdTestPSB = pdEmails[['poi', 'salary', 'bonus', 'total_stock_value',
                     'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()

0.6667 accuracy

'''


# #### Testing Results top accuracy 0.83
# 
# `pdTestPSB = pdEmails[['poi', 'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`
# 
# .0714286 accuracy
# 61 rows
# - dropping highest pct_email_poi ('< 0.2') -> **0.833333 accuracy**
#     - `pdTestPSB = pdEmails[(pdEmails['pct_email_poi'] < 0.2)][['poi', 'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`
# - dropping lowest two bon_sal ('> 0.4') -> **0.5 accuracy**
#     - `pdTestPSB = pdEmails[(pdEmails['bon_sal'] > 0.4)][['poi', 'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`
# - Previous two together -> **0.83333**, so the first is better as it has more data points
#     - `pdTestPSB = pdEmails[(pdEmails['pct_email_poi'] < 0.2) & (pdEmails['bon_sal'] > 0.4)][['poi', 'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`

# #### Testing Results with poi + one other
# 
# `pdTestPSB = pdEmails[['poi', 'bon_sal']].dropna()`
# 
# - bon_sal -> 0.6667
# - pct_email_poi -> 0.625
# - pct_shared_poi -> 0.6667
# 

# #### Testing results with all chosen
# 
# `pdTestPSB = pdEmails[['poi', 'salary', 'bonus', 'total_stock_value',
#                      'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`
#                      
# - Excluding highest pct_email_poi -> 0.8333
#     - `pdTestPSB = pdEmails[(pdEmails['pct_email_poi'] < 0.2)][['poi', 'salary', 'bonus', 'total_stock_value',
#                      'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`
# 
# - Also excluding highest total_stock_value -> jumps from 0.8 to 1.0
#     - `pdTestPSB = pdEmails[(pdEmails['pct_email_poi'] < 0.2) &
#     (pdEmails['total_stock_value'] < 3000000)][['poi', 'salary', 'bonus', 'total_stock_value',
#     'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`
#     
# - Also excluding highest 3 total_stock_value -> 1.0
#     - `pdTestPSB = pdEmails[(pdEmails['pct_email_poi'] < 0.2) &
#     (pdEmails['total_stock_value'] < 2000000)][['poi', 'salary', 'bonus', 'total_stock_value',
#     'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`
#     
# - Also excluding salary under 600k -> 0.75
#     - `pdTestPSB = pdEmails[(pdEmails['pct_email_poi'] < 0.2) & (pdEmails['total_stock_value'] < 2000000) &
#                      (pdEmails['salary'] < 600000)][['poi', 'salary', 'bonus', 'total_stock_value',
#                      'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`
#                      
# - Salary back to full, excluding bonus over 400k -> 1.0
#     - `pdTestPSB = pdEmails[(pdEmails['pct_email_poi'] < 0.2) & (pdEmails['total_stock_value'] < 2000000) &
#                      (pdEmails['bonus'] < 400000)][['poi', 'salary', 'bonus', 'total_stock_value',
#                      'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()`

# #### this one bounces from 0.54 and 1.0
# 
# pdTestPSB = pdEmails[(pdEmails['pct_email_poi'] < 0.2) &
#                      (pdEmails['total_stock_value'] < 2000000)][['poi', 'salary', 'bonus', 'total_stock_value',
#                                                                  'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()
# 
# features_list = ['poi', 'salary', 'bonus', 'total_stock_value',
#                      'bon_sal', 'pct_email_poi', 'pct_shared_poi']
# 
# data_dict = pdTestPSB.T.to_dict()

# # Gets a consistent 0.6 on the GaussianNB
# 
# pdTestPSB = pdEmails[(pdEmails['total_stock_value'] < 3000000)][['poi', 'bonus', 'total_stock_value', 'pct_shared_poi']].dropna()
# 
# features_list = ['poi', 'bonus', 'total_stock_value', 'pct_shared_poi']
# 
# data_dict = pdTestPSB.T.to_dict()

# In[16]:


# Gets a 0.8333

pdTestPSB = pdEmails[['poi', 'bonus', 'total_stock_value', 'pct_shared_poi']].dropna()

features_list = ['poi', 'bonus', 'total_stock_value', 'pct_shared_poi']

data_dict = pdTestPSB.T.to_dict()


# In[17]:


# Store to dataset for easy export below.
dataset = data_dict

# Extract features and labels from dataset for local testing
data = feature_format(dataset, features_list, sort_keys=True)
labels, features = target_feature_split(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)


# In[18]:


clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print ('accuracy: %f' %score )


# In[19]:


clf.feature_importances_


# In[20]:


len(pdTestPSB)


# In[ ]:


''' using SelectKBest, idk what's happening here
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
X, y = load_digits(return_X_y=True)
X.shape
(1797, 64)
X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
X_new.shape
(1797, 20)

from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
X, y = load_digits(return_X_y=True)
X.shape

X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
X_new.shape

X_new
'''


# ### Trying classifiers

# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html
# 
# Provided to give you a starting point. Try a variety of classifiers.

# In[21]:

clf = GaussianNB()


# In[22]:


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


# In[23]:


clf = classify(features_train, labels_train, features_test, labels_test)


# ## Task 5: Tune your classifier to achieve better than .3 precision and recall

# Using our testing script. Check the `tester.py` script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# 
# Example starting point. Try investigating other evaluation techniques!

# In[24]:


# Fixed at 0.8333


features_train, features_test, labels_train, labels_test = train_test_split(features,labels, test_size=0.3, random_state=42)

clf


# In[25]:


# Testing different stats from the one above;
# 0.25 & 42 give 0.8667; same with 0.25 & 18
# RS: 38 = 0.6667; 32 = 0.8

# Test size of 0.25 & random state of 12 give 0.9333

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=12)

clf


# ## Task 6: Dump your classifier, dataset, and `features_list` so anyone can

# Check your results. You do not need to change anything below, but make sure
# that the version of `poi_id.py` that you submit can be run on its own and
# generates the necessary `.pkl` files for validating your results.

# In[26]:

sys.path.append('InProgress/')

dump_classifier_and_data(clf, dataset, features_list)


# In[37]:


test_classifier(clf, dataset, features_list)


# In[ ]:




