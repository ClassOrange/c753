#!/usr/bin/env python
# coding: utf-8

# # Starter Code for the Lesson 17: Final Project

# In[124]:


import sys
sys.path.append("../utils/")

import pickle
from feature_format import feature_format, target_feature_split
from tester import dump_classifier_and_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.naive_bayes import GaussianNB


# In[2]:


#Initial data load

with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


# ## Task 1: Select what features you'll use

# In[3]:


# create pandas data frame; .T is to transpose and use names as rows/features as columns
pdEmails = pd.DataFrame(data_dict).T
pdEmails.head()


# In[4]:


# remove the 'TOTAL' row as we know it's not valid
pdEmails = pdEmails.drop('TOTAL', axis = 0)

pdEmails.dtypes


# In[5]:


# need to be able to work with numerical data properly, but int won't allow NaN
cols = pdEmails.columns
noChange = ['poi', 'email_address']

for col in cols:
    if col not in noChange:
        pdEmails[col] = pdEmails[col].astype('float64')

pdEmails.dtypes


# In[6]:


pdEmails.describe()


# In[7]:


pdEmails.info()


# - 145 people/rows
# - 19 columns/features (21 if index and name are included)
# - Only email addresses and poi have the full 145 entries
#     - Really this is only 'poi' as some email addresses are NaN and that still just shows up as a value
#     - Director fees has 16 entries
#     - Expenses has 94 entries
#     - Loan advances has 3 entries, etc.

# In[8]:


pdEmails[(pdEmails['director_fees']> 1)]


# In[9]:


pdEmails[(pdEmails['deferral_payments'] > 1)]


# Some features may be removed without real further exploration.
# - 'loan_advances' has only 3 values (only 1 POI)
# - 'director_fees' actually really interests me; only two of those people have Enron email addresses, none are POI's, and we have almost no further information on any of them. So although I'm extremely curious about them, I just have nowhere to go with it
# - any of the defer categories; they don't have enough data points and most aren't associated with any POI
# - 'to_messages' and 'from_messages' don't seem to have much bearing on anything by concept alone; raw email volume can be skewed by anything
# 
# Potentially interesting
# - 'other' as I don't know what it is
# - 'expenses'; that seems like something easy to abuse in a company like Enron so I want to look at them

# Not actually running this as the new list in case I still want to use any of those I've discounted as useless in order to create new variables
# 
# pdEmails = pdEmails[['salary', 'total_payments', 'bonus', 'email_address', 'total_stock_value',
#                      'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'other',
#                      'from_this_person_to_poi', 'poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock']]

# In[10]:


pdPOI = pdEmails[(pdEmails['poi'] == True)]


# I'm automatically interested in anyone with a high salary and/or bonus

# In[11]:


sb.scatterplot(data=pdEmails, x='salary', y='bonus', hue='poi');


# In[12]:


pdEmails[(pdEmails['bonus'] > 7000000) | (pdEmails['salary'] > 1000000) & (pdEmails['poi'] == False)]


# 'LAVORATO JOHN J' and 'FREVERT MARK A' have extremely high bonus/salary respectively and neither is a POI

# In[13]:


pdEmails[(pdEmails['bonus'] > 2000000) & (pdEmails['poi'] == False)]


# 'MCMAHON JEFFREY' 'KITCHEN LOUISE' 'LAVORATO JOHN J' (mentioned above) 'WHALLEY LAWRENCE G' 'FALLON JAMES B' 'ALLEN PHILLIP K' are non-POI with bonuses higher than 2 million
# 
# High total payments may be useful as well (or it could elimite the need for salary/bonus as individual comparitors)

# In[14]:


sb.scatterplot(data = pdEmails, x = 'salary', y = 'total_payments', hue = 'poi')


# In[15]:


pdEmails[(pdEmails['total_payments'] > 100000000)]


# Kenneth Lay is that super high one; that tracks. Looking at the same plot but excluding salaries above $600k

# In[16]:


sb.scatterplot(data = pdEmails[(pdEmails['salary'] < 600000)], x = 'salary', y = 'total_payments', hue = 'poi')


# Off the bat, I'm learning nothing here; the distribution of POI vs non-POI in the scope of total payments and salaries doesn't seem to show any sort of pattern.

# In[17]:


# bonus as a percentage of bonus + salary
pdEmails['bon_sal'] = pdEmails['bonus'] / (pdEmails['bonus'] + pdEmails['salary'])


# In[18]:


#df.hist(column='Test1', by='Major')
pdEmails.hist(column = 'bon_sal', by = 'poi')


# If bonuses as a percentage of bonus and salary is an indicator, it's not a very strong one; there definitely does seem to be a difference (POI have a higher percentage), but it doesn't seem that reliable/useful

# In[19]:


sb.scatterplot(data = pdEmails, x = 'salary', y = 'expenses', hue = 'poi')


# In[20]:


pdEmails[(pdEmails['expenses'] > 150000)]


# expenses doesn't look like anything either

# In[21]:


sb.scatterplot(data = pdEmails, x = 'from_poi_to_this_person', y = 'from_this_person_to_poi', hue = 'poi')


# In[22]:


pdEmails[(pdEmails['from_this_person_to_poi'] > 100)]


# Overall here it's showing non-POI sent more emails to POI than POI sent emails to one another; idk what to do with this yet, but it seems it could easily be important

# In[23]:


pdEmails.info()


# In[24]:


# calculating a potential feature: emails sent to/received from POI as percentage of total email sent/received

pdEmails['pct_email_poi'] = (pdEmails['from_poi_to_this_person'] + pdEmails['from_this_person_to_poi']) / (pdEmails['from_messages'] + pdEmails['to_messages'])


# In[25]:


sb.boxplot(data = pdEmails, x = 'poi', y = 'pct_email_poi')


# The percentage of a person's total emails that had to do with a POI tends to be lower with a non-POI, but then there are a handful with far higher interaction by %. So that's either a data issue/irrelevant or maybe those are people I should be looking at more as potential POI

# In[26]:


pdEmails[(pdEmails['pct_email_poi'] > .13)]


# In[27]:


pdEmails['pct_shared_poi'] = pdEmails['shared_receipt_with_poi'] / pdEmails['to_messages']


# In[28]:


sb.boxplot(data = pdEmails, x = 'poi', y = 'pct_shared_poi')


# In[29]:


sb.scatterplot(data = pdEmails, x = 'pct_shared_poi', y = 'exercised_stock_options', hue = 'poi')


# The % of shared receipts with POI is a bit iffy with a lot of things; various monetary things like long term incentive, total stock value, restricted stock, and exercised stock do cluster a bit higher with a higher shared %, BUT the highest monetary values fall closer to the center of shared POI%, and the rise is not significant

# In[30]:


pdEmails[(pdEmails['pct_shared_poi'] > .7) & (pdEmails['poi'] == False)][['pct_shared_poi', 'salary', 'bonus', 'poi']]


# This seems very intriguing. Plenty of people who are not currently in the POI box have very high percentages for shared receipts with POI, as well as high salaries and bonuses. Many of these names are also familiar from previous subsets wherein I filtered down to the suspicious elements.
# 
# An issue, however, is clearly the data isn't super accurate; Ben 'GLISAN JR BEN F' shows a percentage of over 100 for this new feature. While there could be a good reason for a percentage over 100 in some instances with certain data, it doesn't make sense in this case; the amount received of a subset of emails should NOT surpass the total amount of emails received.

# In[31]:


sb.scatterplot(data = pdEmails[(pdEmails['salary'] < 800000) & (pdEmails['long_term_incentive'] < 3000000)], x = 'salary', y = 'long_term_incentive', hue = 'poi')


# In[32]:


sb.scatterplot(data = pdEmails, x = 'long_term_incentive', y = 'pct_shared_poi', hue = 'poi')


# In[33]:


pdEmails.columns


# Removing
# - 'email_address' as it doesn't help any further with this type of analysis
# - 'total_payments' 

# Reiterating from above, removing:
# 'to_messages',
# 'deferral_payments',
# 'loan_advances',
# 'restricted_stock_deferred',
# 'deferred_income',
# 'from_messages',
# 'director_fees'
# 
# Now also removing:
# 'total_payments',
# 'email_address',
# 'expenses'
# 'from_this_person_to_poi'
# 'from_poi_to_this_person'
# 'shared_receipt_with_poi'
# 'exercised_stock_options'
# 'restricted_stock'
# 'other'
# 'long_term_incentive'

# #### Creating testing data frame and matching features list

# In[109]:


# current features after additions; pare this down more based on charts and such above

pdEmailsCut = pdEmails[['poi', 'salary', 'bonus', 'total_stock_value',
                     'bon_sal', 'pct_email_poi', 'pct_shared_poi']]

pdChosenNoNA = pdEmailsCut.dropna()


# In[111]:


len(pdChosenNoNA)


# In[35]:


# The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_stock_value',
                     'bon_sal', 'pct_email_poi', 'pct_shared_poi']


# In[36]:


# count of POI in pared down chosen set

len(pdChosenNoNA[(pdChosenNoNA['poi'] == True)])


# In[37]:


len(pdChosenNoNA)


# ## Task 2: Remove outliers

# #### Trying to do more testing re: features, post new data frame

# In[38]:


sb.scatterplot(data = pdChosenNoNA, x = 'salary', y = 'bonus', hue = 'poi')


# In[39]:


sb.histplot(data = pdChosenNoNA, x = 'bon_sal', hue = 'poi')


# In[ ]:





# ## Task 3: Create new feature(s)

# #### Testing features using automated stuff

# In[40]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[42]:


pdTestPSB


# In[117]:


# best outcome from test, but removes a lot of data; other options are in the running_tests page, in case
# this one is only 36 length

pdTestPSB = pdEmails[(pdEmails['pct_email_poi'] < 0.2) &
                     (pdEmails['total_stock_value'] < 2000000)][['poi', 'salary', 'bonus', 'total_stock_value',
                                                                 'bon_sal', 'pct_email_poi', 'pct_shared_poi']].dropna()

features_list = ['poi', 'salary', 'bonus', 'total_stock_value', 'bon_sal', 'pct_email_poi', 'pct_shared_poi']


# In[118]:


# back to dictionary data_dict
# Have to use chopped up one because NaNs
data_dict = pdTestPSB.T.to_dict()


# In[119]:


# Store to my_dataset for easy export below.
my_dataset = data_dict

# Extract features and labels from dataset for local testing
data = feature_format(my_dataset, features_list, sort_keys=True)
labels, features = target_feature_split(data)

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=42)


# In[122]:


clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
pred= clf.predict(features_test)
print ('accuracy: %f' %score )


# In[123]:


len(pdTestPSB)


# In[72]:


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


# #### other stuff

# In[ ]:





# In[ ]:





# In[ ]:





# ## Task 4: Try a varity of classifiers

# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html
# 
# Provided to give you a starting point. Try a variety of classifiers.

# In[47]:


from sklearn.naive_bayes import GaussianNB


clf = GaussianNB()


# ## Task 5: Tune your classifier to achieve better than .3 precision and recall

# Using our testing script. Check the `tester.py` script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# 
# Example starting point. Try investigating other evaluation techniques!

# In[48]:


from sklearn.model_selection import train_test_split


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


# ## Task 6: Dump your classifier, dataset, and `features_list` so anyone can

# Check your results. You do not need to change anything below, but make sure
# that the version of `poi_id.py` that you submit can be run on its own and
# generates the necessary `.pkl` files for validating your results.

# In[49]:


dump_classifier_and_data(clf, my_dataset, features_list)

