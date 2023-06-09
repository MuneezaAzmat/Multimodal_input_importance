# Synthetic
This repository contains Python code for a validation of our multimodal input importance framework using synthetic data. link to paper [link]. 
The code is organized into several Python scripts

## 00_Data_preprocessing.py
This script is responsible for data preprocessing. It loads two image datasets, 'organamnist.npz' and 'organcmnist.npz', and pre-processes them (normalize, reshape).
The script also defines an autoencoder model which is trained to extract deep features from the images. These extracted features are used in the syntehtic decision functions for generating ground truth labels. The script alse generates synthetic (tabular) clinical data.

## 01_GT_gen.py
This script generates ground truth labels and importances for the input features. 
It loads the previously save image features and tabular inputs and calculates the importance of each feature. 
The script also generates labels for the data based on a threshold that generates balances dataset. 
The importance and labels are saved for future use.

## 02_Multimodal_training.py
This script is responsible for training a multimodal model. 
It defines a custom dataset class for loading the multimodal data and labels. 
It also defines an encoder for image data and a fused network for combining the image and tabular data. 
The script trains the model using the Adam optimizer and CrossEntropyLoss. 
The trained models are saved for future use.
The synthetic decision function and its derivative are custom inputs and can be changed to experiment with a diverse set of classification problems.

## 03_GRAD.py
Uses gradient-based method to compute the importance of features. 
It calculates the gradients of the model's output with respect to the inputs, which are interpreted as the importance of each feature.
The script also normalizes the importance values and plots the importance of different modalities.

## 04_PERM.py
Computes the Permutation Importance of features. 
Permutation Importance is calculated by permuting the values of each feature and measuring the decrease in the model's performance.
The script also plots the importance of different modalities.

## 05_LIME.py
Uses the LIME (Local Interpretable Model-agnostic Explanations) method to explain the predictions of the model. 
LIME generates a set of perturbations around a data point, gets the model's predictions for these perturbations, 
and then fits a local linear model to approximate the model's behavior in the vicinity of the data point. 
The coefficients of the linear model are interpreted as the importance of each feature. The script also plots the importance of different modalities.

## 06_SHAP.py
Uses the SHAP (SHapley Additive exPlanations) method to explain the predictions of the model. 
SHAP computes the Shapley values for each feature, which represent the average marginal contribution of the feature to the prediction for all possible subsets of features. 
The script also plots the importance of different modalities.These scripts provide different ways to understand the importance of each feature in the 
model's predictions, which can be useful for interpreting the model's behavior and diagnosing potential issues.

## 07_Plots.py 
Loads all saved normalized importance estimates and plots bar-charts for comparison.
