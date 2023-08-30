#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


def evaluation_metrics(model, X_test, y_test, model_name, labels):
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    print("Classifier: {}".format(model_name))

    #===================================================================================
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: {:.3f}".format(accuracy))
    print("\n")

    #===================================================================================
    # Calculate Recall
    recall = recall_score(y_test, y_pred)

    #===================================================================================
    # Calculate Precision
    precision = precision_score(y_test, y_pred)
    
    #===================================================================================
    # Calculate F1 score
    F1 = f1_score(y_test, y_pred)

    #===================================================================================
    # Generate classification report
    class_report = classification_report(y_test, y_pred, target_names=labels)
    print("Classification Report:\n", class_report)
    print("\n")

    #===================================================================================
    # Display confusion matrix using ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, y_pred, normalize="all")
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()
    print("\n")

    #===================================================================================
    # Get the probabilities of the positive class (label 1) to calculate ROC AUC
    y_probs = model.predict_proba(X_test)[:, 1]

    # Calculate ROC AUC score
    roc_auc_Score = roc_auc_score(y_test, y_probs)
    print("ROC AUC Score: {:.2f}".format(roc_auc_Score))
    print("\n")

    # Generate ROC curve using RocCurveDisplay
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_Score, estimator_name=model_name)
    roc_display.plot()
    plt.title('ROC Curve')
    plt.show()


    return accuracy, recall, precision, F1, roc_auc_Score, fpr, tpr


# In[ ]:


def scores_table(model_names, accuracy_scores, precision_scores, recall_scores, f1scores, roc_auc_scores, title):
    """
    Print a table with model performance scores.

    Parameters:
        model_names : List of model names.
        accuracy_scores : List of accuracy scores corresponding to each model.
        recall_scores : List of recall scores corresponding to each model.
        roc_auc_scores : List of ROC AUC scores corresponding to each model.
    """
    # Format scores to two decimal places
    accuracy_scores = [f"{score:.4f}" for score in accuracy_scores]
    precision_scores = [f"{score:.4f}" for score in precision_scores]
    recall_scores = [f"{score:.4f}" for score in recall_scores]
    f1scores = [f"{score:.4f}" for score in f1scores]
    roc_auc_scores = [f"{score:.4f}" for score in roc_auc_scores]
    
    # create table    
    data = list(zip(model_names, accuracy_scores, precision_scores, recall_scores, f1scores, roc_auc_scores))
    headers = ['Models', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    table = tabulate(data, headers=headers, tablefmt="grid")
    print(f"{title}\n")
    print(table)


# In[ ]:


def plot_barchart(model_names, scores, score_name, plot_title):
    """
    Create a bar chart for different models.

    Parameters:
        model_names : List of model names.
        scores : List of accuracy/recall/precision scores corresponding to each model.
    """
    
    plt.bar(model_names, scores, color=sns.color_palette())
    plt.xlabel('Models')
    plt.ylabel(score_name)
    plt.title(plot_title)
    plt.xticks(rotation=90)
    plt.show()


# In[ ]:


def plot_roc_auc(model_names, fpr_arr, tpr_arr, roc_auc_scores, plot_title):
    """
    Plot ROC AUC curves for different models.

    Parameters:
        model_names : List of model names.
        fpr_arr : List of FPR arrays corresponding to each model.
        tpr_arr : List of TPR arrays corresponding to each model.
        roc_auc_scores : List of ROC AUC scores corresponding to each model.
    """
   
    for model_name, fpr, tpr, roc_auc in zip(model_names, fpr_arr, tpr_arr, roc_auc_scores):
        plt.plot(fpr,tpr,label=f"{model_name} (AUC = {roc_auc:.2f})")    

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(plot_title)
    plt.legend()
    plt.show()


# In[ ]:

def grid_search_cv(X_train, y_train, models, param_grids, cv=3):
    best_models = []
    best_scores = []

    # Initialize StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    for model, param_grid in zip(models, param_grids):
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=3)

        grid_search.fit(X_train_scaled, y_train)

        best_models.append(grid_search.best_estimator_)
        best_scores.append(grid_search.best_score_)

    return best_models, best_scores

