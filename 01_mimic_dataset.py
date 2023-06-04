import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class PathologyDataset(Dataset):
    def __init__(self, csv_path, img_dir, tab_feat_names, pathology):
        # Initialize the dataset
        self.df = pd.read_csv(csv_path)  # Read the CSV file
        self.img_dir = img_dir  # Directory containing the images
        self.tab_feat_names = tab_feat_names  # Names of tabular features
        self.pathology = pathology  # Name of the pathology label

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.df)

    def __getitem__(self, idx):
        # Get a sample from the dataset at a specific index

        # Extract tabular features and reshape into a 1D array
        tabular = self.df.loc[idx][self.tab_feat_names].values.astype(float).reshape(-1)

        # Extract the pathology label and reshape into a 1D array
        label = np.array(self.df.loc[idx][self.pathology]).reshape(-1)

        # Construct the image file name from the path and load the image as a numpy array
        image_name = self.df.loc[idx]['path_to_image'].replace('/', '_')[10:]
        image = np.load(os.path.join(self.img_dir, image_name+'.npy'))

        # Convert tabular features, image, and label to Torch tensors
        tensor_tabular = torch.Tensor(tabular).float()
        tensor_image = torch.Tensor(image).float()
        tensor_label = torch.Tensor(label).float()

        # Return a tuple of Torch tensors representing the image, tabular features, and label
        return (tensor_image, tensor_tabular, tensor_label)
