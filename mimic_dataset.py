import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset

class PathologyDataset(Dataset):
    def __init__(self, csv_path, img_dir, tab_feat_names, pathology):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tab_feat_names = tab_feat_names
        self.pathology = pathology

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        tabular = self.df.loc[idx][self.tab_feat_names].values.astype(float).reshape(-1)
        label = np.array(self.df.loc[idx][self.pathology]).reshape(-1)  
        image_name = self.df.loc[idx]['path_to_image'].replace('/', '_')[10:]
        image = np.load(os.path.join(self.img_dir,image_name+'.npy'))

        tensor_tabular = torch.Tensor(tabular).float()
        tensor_image = torch.Tensor(image).float()
        tensor_label = torch.Tensor(label).float()
        
        return (tensor_image, tensor_tabular, tensor_label)

