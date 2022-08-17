# Credit Risk Analysis


## Overview

Calculation risk in the financial industy can sometimes be a messy process.  With machine learning, we can use multiple techniques to build credit risk models that continue to improve decision making as the model gains more and more data.  In this analysis, with the help of data from LendingClub, a peer-to-peer lending platform, we will use multiple techniques to predict the best outcome of credit risk for this platform. 

## Resources

- Data: LoanStats_2019Q1.csv

- Languages: Python

- Machine Learning:
  -sklearn
  -imblearn
  -scikit-learn

## Results

Given a dataset with 66,000+ examples, it was important to test multiple scenarios for detecting credit risk.  In these scenarios, we attempted to build a model that could accurately predict credit risk, the likelihood of a candidate being a 'good credit risk', a good candidate for lending, or a 'bad credit risk', a candidate that is not creditworthy.    Based on the shape of the data, we can see that the dataset is imbalanced.  In order to consider the risk for this, multiple models were trained to test for the likelihood of predicting good vs bad credit candidates. 

### Naive Random Sampling

In our first model, we used a Random Oversampling method from the imblearn module.  When considering the same of our data, we can see that the number of good applicants signifcantly outways the number of bad applicants.  In this case, we want to balance our model in order to determine achieve a higher degree of accuracy in the model, otherwise the model will perform well with an imbalance, which is likely a false result. After sampling the data, we trained the model with Logistic Regression to predict the outcomes of good applicants vs bad applicants from this data.  Using this model, we found:

  - A balanced accuracy score of approximately 65%.  This model was able to predict true positives 65% of the time.  
  - This model was sensitive to accurately predicting high_risk and low_risk applicants.


### SMOTE Oversampling

In our second model, we used a synthetic minority oversampling technique (SMOTE).  In this model, the minority class (bad loan applicants) is increased to the size of the majority class and tested.  After sampling the data using SMOTE, we trained the model with Logistic Regression to predict the outcomes from this data.  Using this model we found:

  - A balanced accuracy score of approximately 65% as well.  This model model was able to predict true positives 65% of the time.  
  - This model was also sensitive to accurately predicting high_risk and low_risk applicants.  

### Undersampling

In our third model, we used a method of Random Undersampling with the ClusterCentroids resampler.  In this model, we identify clusters in the majority class (good loan applicants), maps those clusters, and finally undersamples the majority class be more balanced with the minority class.  This model involves a loss of data from the majority class, thus we anticipate the model to have the least accurate results.  We then trained a model with Logistic Regression to predict the accuracy of the model. We found that:

  - Undersampling produced worse results than the first two models, only predicting true positives 53% of the time.
  - The model was able to predict low risk well, but not as well as both oversampling techniques.  

### Combination Sampling

In our fourth model, we used a combination method of Undersampling and Oversampling using SMOTEENN.  This method combines SMOTE with ENN, which oversamples the minority class and cleans the resulting data with undersampling.  When datapoints overlap, the datapoints are dropped, creating a finer line between classes.  From this sampling, we use a Logistic Regression model to train our data.  Once the data is trained and tested, we found:

  - The fourth model was on par with the first two models, predicting true positives 63% of time time.
  - The model was must better at being able to predict low risk applicants than higher risk applicants, based on an F1 score of .71 v .02.


# Summary

Overall, the models appears to produce less than stellar results.  When considering credit risk, it appears to be more important for the test to have high sensitivity, meaning it is better for the model to accurately predict the number of bad candidates in order to mitigate risk.

From this dataset, Combination sampling appears to be the most reliable due to it's tradeoffs with oversampling and undersampling a large imbalanced dataset.  It had a higher accuracy score, and was slightly better at assessing sensitivity to high_risk applicants (recall = .71, a measure for sensitivity).

Based on the data, undersampling was expected to perform the worst among the datasets, and the summary reflects this assumption.  This model had a accuracy score under 60%, and had worse sensitivity scores than the other models.  

