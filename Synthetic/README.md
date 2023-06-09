# Synthetic
This repository contains Python code for a validation of our multimodal input importance framework using synthetic data. link to paper [link]. 
The code is organized into several Python scripts

## 00_Data_preprocessing.py
This script is responsible for data preprocessing. It loads two datasets, 'organamnist.npz' and 'organcmnist.npz',
and applies transformations to convert them into PyTorch tensors. The datasets are normalized and reshaped for further processing.
The script also defines an encoder and a decoder for an autoencoder model. The autoencoder is trained, and the trained models are saved for future use.
The script also extracts deep features from the images using the trained encoder and generates synthetic clinical data.

## 01_GT_gen.py
This script generates ground truth importance for the features. 
It loads the previously saved features and calculates the importance of each feature. 
The script also generates labels for the data based on a predefined condition. 
The importance and labels are saved for future use.

## 02_Multimodal_training.py
This script is responsible for training a multimodal model. 
It defines a custom dataset class for loading the data and labels. 
It also defines an encoder for image data and a fused network for combining the image and tabular data. 
The script trains the model using the Adam optimizer and CrossEntropyLoss. 
The trained models are saved for future use.

## 03_GRAD.py
This script was expected to contain the implementation of Grad-CAM for visualizing the importance of different regions in the input images. 
However, the file was not found in the repository.The repository provides a comprehensive workflow for working with multimodal data, 
training a model on it, and analyzing the importance of different features. 
It can be a valuable resource for anyone interested in multimodal data analysis and feature importance visualization.


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
