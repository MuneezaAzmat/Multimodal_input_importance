# Hybrid Fusion Multimodal Neural Network for Pathology Classification with Feature Importance and Interpretability

This repository contains code for a hybrid fusion multimodal neural network trained for a classification task, specifically focused on predicting the probability of the presence of pathology based on a HIAM Study. The model takes input from different modalities, such as images and tabular data, and provides a classification prediction. In addition, the repository provides functionality for multimodal input importance analysis, enhancing the model's interpretability.

## Model Architecture

The neural network architecture used in this repository is a PyTorch hybrid fusion multimodal neural network. The model combines an image encoder, a tabular encoder, and a fusion network to learn representations from both modalities and make predictions. The fusion network integrates the encoded features from both modalities to produce the final classification output.

## End-to-End Training

The multimodal model is trained in an end-to-end fashion. This means that both the image encoder and the tabular encoder are trained simultaneously with the fusion network. By training all components together, the model can learn to extract meaningful representations from both image and tabular data and optimize the fusion process.

## Multimodal Input Importance

After training the model, the repository provides functionality to perform multimodal input importance analysis. This analysis aims to determine the importance of different modalities in making predictions. Four methods are implemented for this purpose:

1. Gradient Importance: This method computes the gradient of the model's output with respect to the input modalities to understand their impact on the predictions.
2. Permutation Importance: By randomly permuting the values of individual modalities, this method assesses the change in the model's performance to measure the importance of each modality.
3. Shapley Values: Shapley values assign importance scores to each modality by considering all possible combinations and permutations of modalities in the model.
4. LIME (Local Interpretable Model-Agnostic Explanations): LIME provides local explanations for the model's predictions by fitting an interpretable model to the predictions of the multimodal model.
5. AGG: This method returns an aggregate of results from all above methods.

The feature importance results are returned at both the modality level and the input level, allowing a detailed analysis of the model's decision-making process.

## Data Requirements

The code in this repository assumes the availability of preprocessed multimodal data. The multimodal data consists of images resized to 224x224 pixels and tabular data. The image data should be stored in a suitable format (e.g., JPEG or PNG files), while the tabular data should be stored in CSV format.

In addition to the data files, a CSV file mapping the tabular data to the corresponding images is required. This mapping allows the code to associate the correct modalities during training and inference.

## Usage

To use this repository, follow these steps:

1. Ensure you have the required multimodal data in the specified format (images resized to 224x224 pixels and tabular data in CSV format).
2. Prepare a CSV file that maps the tabular data to the corresponding images.
3. Use the provided data loader to load the multimodal data and create suitable data structures for training and evaluation.
4. Train the hybrid fusion multimodal neural network by running the appropriate training script.
5. Once the model is trained, utilize the multimodal input importance analysis methods to understand the importance of different modalities in the model's predictions.

Please refer to the code documentation and comments for more detailed information on how to use the provided functionalities.

## Contributors

This repository was created and is maintained by Muneeza Azmat. Contributions and suggestions are welcome.

If you encounter any issues or have questions, please open an issue in the repository or contact azmatmun@msu.edu
