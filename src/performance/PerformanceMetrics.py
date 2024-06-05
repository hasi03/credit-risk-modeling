# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, recall_score, precision_score
from sklearn_evaluation.plot import ConfusionMatrix
import xgboost as xgb


class ModelEvaluator:

    def __init__(self, model, model_type='xgb'):
        self.model = model
        self.model_type = model_type
        self.y_pred_proba = None
        self.y_pred = None

    def predict(self, X_test, y_test):
        if self.model_type == 'xgb':
            d_test = xgb.DMatrix(X_test, label=y_test)
            self.y_pred_proba = self.model.predict(d_test)
        elif self.model_type == 'xgb_classifier':
            self.y_pred_proba = self.model.predict(X_test)
        else:
            raise ValueError('Invalid model type')

    def apply_threshold(self, new_threshold=0.5):
        self.y_pred = (self.y_pred_proba >= new_threshold).astype(int)

    
    def plot_roc(self, y_test):
        if self.y_pred_proba is None:
            raise ValueError('Run predict() first to get y_pred_proba')
        
        fpr, tpr, thresholds = roc_curve(y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.text(0.6, 0.2, 'AUC: ' + str(round(roc_auc, 2)), fontsize=12)
        
        return fig

    def get_metrics(self, y_test, new_threshold=0.5):
        if self.y_pred_proba is None:
            raise ValueError('Run predict() first to get y_pred_proba')
        
        self.apply_threshold(new_threshold)
        
        conf_matrix = confusion_matrix(y_test, self.y_pred)
        recall = recall_score(y_test, self.y_pred)
        precision = precision_score(y_test, self.y_pred)
        f1 = f1_score(y_test, self.y_pred)
        
        return conf_matrix, recall, precision, f1
    

    def optimal_threshold(self, y_test):

        # calculate the fpr and tpr
        fpr, tpr, thresholds = roc_curve(y_test, self.y_pred_proba)

        '''Returns the optimal threshold for the roc curve.'''
        # maximize the sum of TPR and 1 - FPR
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        return optimal_threshold, optimal_idx
    




"""
# function for plotting the roc curve
def plot_roc(model, X_test, y_test, model_type = 'xgb'):
    '''Returns the roc curve for the model.'''

    # predict the probabilities
    if model_type == 'xgb':
        d_test = xgb.DMatrix(X_test, label=y_test)
        y_pred_proba = model.predict_proba(d_test)

    elif model_type == 'xgb_classifier':
        y_pred_proba = model.predict(X_test)
    
    else:
        return 'Invalid model type'

    # calculate the fpr and tpr
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # calculate the auc
    roc_auc = auc(fpr, tpr)
    
    # plot the roc curve
    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')

    # set labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # set title
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # show the auc score
    plt.text(0.6, 0.2, 'AUC: '+str(round(roc_auc, 2)), fontsize=12)
    
    return fig





def metrics(model, X_test, y_test, new_threshold = 0.5, model_type = 'xgb'):
    '''Returns the confusion matrix, recall, precision, and f1 score for the model.'''

     # predict the probabilities
    if model_type == 'xgb':
        d_test = xgb.DMatrix(X_test, label=y_test)
        y_pred = model.predict_proba(d_test)

    elif model_type == 'xgb_classifier':
        y_pred = model.predict(X_test)
    
    else:
        return 'Invalid model type'

    # Apply the new threshold to determine class labels
    y_pred = (y_pred >= new_threshold).astype(int)

    # calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # calculate the recall
    recall = recall_score(y_test, y_pred)

    # calculate the precision
    precision = precision_score(y_test, y_pred)

    # calculate the f1 score
    f1 = f1_score(y_test, y_pred)

    # return the metrics
    return conf_matrix, recall, precision, f1


def optimal_threshold(model, X_test, y_test):

    y_pred_proba = model.predict(X_test)

    # calculate the fpr and tpr
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    '''Returns the optimal threshold for the roc curve.'''
    # maximize the sum of TPR and 1 - FPR
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold, optimal_idx
"""