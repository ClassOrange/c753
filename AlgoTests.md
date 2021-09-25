### DecisionTreeClassifier(no tweaking)

#### 'poi', 'bonus', 'total_stock_value' < 2000000, 'pct_shared_poi' < 0.2

Accuracy: 0.76425	Precision: 0.53532	Recall: 0.43200	F1: 0.47814
F2: 0.44934 Total predictions: 4000	True positives:  432
False positives:  375	False negatives:  568	True negatives: 2625

DecisionTreeClassifier(small inline run)
accuracy: 0.777778
[0.34603896 0.14464286 0.50931818]

GaussianNB
Out[64]: 0.7777777777777778

#### 'poi', 'bonus', 'total_stock_value', 'pct_shared_poi' < 0.2

Accuracy: 0.76333	Precision: 0.33796	Recall: 0.43800	F1: 0.38153
F2: 0.41352	Total predictions: 6000	True positives:  438
False positives:  858	False negatives:  562	True negatives: 4142

DecisionTreeClassifier(small inline run)
accuracy: 1.000000
[0.49050779 0.14369501 0.36579719]

GaussianNB
Out[74]: 0.9333333333333333

#### 'poi', 'bonus', 'total_stock_value' < 2000000, 'pct_shared_poi'

Accuracy: 0.76500	Precision: 0.53741	Recall: 0.43100	F1: 0.47836
F2: 0.44877	Total predictions: 4000	True positives:  431
False positives:  371	False negatives:  569	True negatives: 2629

DecisionTreeClassifier(small inline run)
accuracy: 0.800000
[0.46390909 0.18409091 0.352]

GaussianNB
Out[82]: 0.8

## Best: 'poi', 'bonus', 'total_stock_value' < 2000000, 'pct_shared_poi'

### DTC(min_samples_split = X)

##### 0.3 -> Accuracy: 0.68525	Precision: 0.27160	Recall: 0.15400
##### 0.1 -> Accuracy: 0.75950	Precision: 0.52375	Recall: 0.41900
##### 0.05 -> Accuracy: 0.76525	Precision: 0.53789	Recall: 0.43300

### DTC(min_weight_fraction_leaf = X)

##### 0.04 -> Accuracy: 0.77425	Precision: 0.56510	Recall: 0.42100
##### 0.06 -> Accuracy: 0.77300	Precision: 0.56183	Recall: 0.41800
##### 0.045 -> Accuracy: 0.77425 Precision: 0.56475	Recall: 0.42300
##### 0.05 -> Accuracy: 0.77425	Precision: 0.56510	Recall: 0.42100

### DTC(min_impurity_decrease = 0.01)

##### 0.01 -> Accuracy: 0.77675	Precision: 0.57220	Recall: 0.42400


