## Machine Learning - Enron Investigation

#### Project Goal

The desired outcome of working with the Enron data is to identify employees beyond the initial Person of Interest (POI) list who may have been involved in the various financial crimes committed. The data includes information for each of 145 employees/people directly associated with those employees; bonuses received, salaries, metadata regarding emails (counts sent/received, counts sent/received associated with those on the POI list), stock values, and more. Additionally, there is a massive collection of the actual emails (from which some of this data was derived), but a full text analysis is a bit beyond the scope of this project. What I'll be doing in particular is looking for patterns in different categories between those on the POI list and those not included. Finding any such relations could in theory lead to a predictor algorithm, automating the identification process.

The data set has a handful of outliers, depending on which feature is being referenced. Many of the monetary value features do correlate between outliers (the same people are often outliers in more than one money measurement, such as having an extremely high salary and bonus). There are also missing values from most, if not all rows; I'm dealing with all this somewhat non-standardized. I only have 58 complete rows of data across the features I'm looking at, so ideally I can use more of the rows when I'm looking at fewer features, but that remains to be seen. Outliers are honestly situational based on tests; sometimes I may want to keep them in but most likely I'll usually just cut them.

- 145 people/rows (146 less the aggregate row in the set)
- 19 columns/features (21 if index and name are included)
- Only email addresses and poi have the full 145 entries
    - Really this is only 'poi' as some email addresses are NaN and that still just shows up as a value
    - Director fees has 16 entries
    - Expenses has 94 entries
    - Loan advances has only 3 entries, etc.
- Worst case scenario of my working data (all of my chosen features with no null values):
    - 58 people/rows
    - 7 features
    - 14 POI
    
#### Features Details

- 'poi'
    - Person of interest; boolean just indicating whether or not a person is currently a person of interest
- 'salary'
    - Employee's salary
- 'bonus'
    - Value of bonus received by the employee
- 'total_stock_value'
    - Value of employee's stock holdings
- 'bon_sal'
    - *Newly created* - A percentage measuring bonus as a percentage of the sum of bonus and salary. This seemed like it could be useful as it makes more sense to compare a number conditionally (within the scope of the related salary) instead of to all the bonuses, regardless of context
- 'pct_email_poi'
    - *Newly created* - The sum of a person's emails sent to and received from those on the POI list divided by the sum of that person's total sent and received emails. As with both the features preceding and following, it made more sense to me to put these numbers in context. 10 emails to a POI when the person's total emails sent is only 11 is far more significant than 10 sent to a POI out of a total of 1000.
- 'pct_shared_poi'
    - *Newly created* - Pretty much the same situation as the previous feature, but this is the count of emails sent to this person as well as at least one POI divided by that person's total emails received

Using the following for features with the pared down data frame, suited to hold all 7 of my features with no NA, the accuracy from the Decision Tree showed 1.0, so I'm sure there's something obvious causing that that I'm just not understanding at the moment. Using a larger dataset, only removing as much necessary for those 3 features to be present without any NA brought that down to 0.67, 0.55, 0.68

`features_list = ['poi', 'salary', 'bon_sal']`

Testing more with the Decision Tree (after choosing the features via manual analysis w/charts and such) brought me to my hopefully final decision: 

`pdTestPSB = pdEmails[['poi', 'bonus', 'total_stock_value' < 2000000,
                      'pct_shared_poi']].dropna()`

Accuracy & importances output:
`accuracy: 0.933333`
`array([0.34789239 0.27570628 0.37640133])`
                     
Essentially I added back all my 7 features, lopped off some lowest values, but then I saw the importance of a couple features at 0, so I pared down from there. My accuracy is showing between 0.8 and 1.0 (alternating between them)

#### Algorithm Used?

Decision Tree is what I went with ultimately (for the tester.py script); Naive Bayes generally gave the same accuracy score as my initial DT tests, and wasn't much different from the more in-depth ones

#### Parameter Tuning

It's difficult to qualify all tuning together, as it's all very specialized; in general, tuning the algorith allows certain thresholds and methods to be tweaked in order to properly utilize the data. Compared features may need to be limited if there are too many of them, or important but less prevalent features may be overlooked if they're not given more weight
- A low value for min_samples_split helped a tiny bit
- I expected min_samples_leaf to go pretty poorly and it sure did
- Switching the method to 'random' from 'best' brought the quality slightly down
- min_weight_fraction_leaf set at around 0.045 helps a bit
- a value of min_impurity_decrease barely helps, but it does help

#### Validation

First, the method I used for validation in this case is the Stratified Shuffle Split provided with the starter code. The gist is the data is shuffled then split into groups that try to keep the identifier labels (POI: True) even between training and testing. Then this is done 1000 times, at least in this case. Without validation, the numbers could really mean anything; when I was running individual quick tests for many of my feature combinations, those numbers sometimes fluctuated entirely between 0.5 and 1 as the accuracy. Ensuring the features are prominent in both the groups means you won't flood one, ie you won't wind up training with all the POI then find yourself unable to validate. In running this many times, we heavily cut down the unsurety of variance.

#### Evaluation

- Accuracy (0.7765) - accuracy is literally just as it sounds. It's the total of accurate predictions (True positive/negative) divided by the total count of predictions, resulting in a percentage
- Precision (0.57162) - true positives divided by the total predicted positives; basically this is to predict whether an approval/risk of some sort will actually result in a positive outcome
- Recall (0.423) - true positives divided by total actual positives (so false negatives included), used more frequently in scenarios wherein it's more dangerous/harmful to miss something than to be wrong