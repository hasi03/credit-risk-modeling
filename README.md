# credit-risk-modeling
Predicting the probability of an applicant paying back the loan.This repository aims to analyze data from different types of personal loans and apply machine learning algorithms to develop a credit risk predictor. 



## 📈 Goal
Predict whether a loan application should be approved or not based on the probability of credit default. I use the following models:
- Random Forest
- XGBoost with **Incremental Learning**


## Installing Required Packages

To install all the required python packages run the following code on linux terminal. 

```bash
  pip install -r requirements.txt
```

  
## 📊 Exploratory Data Analysis

### Data

[Kaggle](https://www.kaggle.com/competitions/home-credit-default-risk/data)

The dataset represents over 300 k personal home loans.

- Each row represents one loan.


### Data Preprocessing

Data preprocessing is an essential step in preparing the data for analysis and modeling. It involves transforming the raw data into a format that is suitable for machine learning algorithms. In this project, the following data preprocessing steps were performed:

1. Handling missing values: Missing values in the dataset were identified and handled using techniques such as imputation or deletion.

2. Handling Outliers: Used Z-score to identify the outliers in the numerical features. Call the module
   ```bash
  from feature_engineering import outliers
  ``` 


### Class Distribution Analysis

 We checked for class imbalance for the target variable.



## Feature Engineering

You can find all the processes we implemented in this section, in _feature-engeering.ipynb_. 


## Data Encoding

First, we used **One-Hot** encoding for cateogorical data that does not have a hierarchical structure. 
Other categorical data with a hierarch, I implemented _Ordinal Encoding_. 


### Feature Selection

We applied two alogrithms seperately to select important features 
- Recursive Feature Elimination (RFE)
- Univariate Feature Selection : ANOVA F-value
- Information Value(IV) and Weight of evidence (WoE)
- Correlation
- Variance Threshold
- Boruta 

## Results

### Performances

We used ROC-auc score as the main metric to evaluate the performance of a model.

To find the best threshold we caluclated the threshold for the maximum of **Young's J Statistic**. 

### Feature Importance 

We utilized 3 different methods to evaluate feature importance.
- Scikit-learn's Feature importance: averaging the decrease in impurity over trees
- Permutation Feature Importance: based on how random re-shuffling of each perdictor influences the model performances. 
- SHAP


## Conclusion

- Best model achieved a **0.66 ROC-auc** score.   
