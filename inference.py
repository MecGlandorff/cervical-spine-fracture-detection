import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import timm

# Program: Cervical fracture detection of c1-c7, inference
# Author: Mec Glandorff
# Version: 1.0
# Description:  This script is the inference part of the project of cervical fracture detection of c1-c7.
#               It uses a cnn backbone with attention layers and a lstm for sequence modeling. 



#############
#Config class
#############

class Config:
    data_dir = "data/masks"
    image_size = 224
    n_per_slice = 15
    in_chans = 6
    out_dim = 1

    # Maybe change in future ?
    batch_size = 6
    num_workers = 4

    # I tested other backbones from efficientnet and similar models. They worked, in the future maybe switch to swin transformer and do architecture
    # overhaul
    backbone = 'tf_efficientnetv2_s_in21ft1k'



################################
# Model definition 
################################
class TimmModelWithAttention(nn.Module):
    def __init__(self, backbone, pretrained=False): # Set pretrained to false because for inference we will load our own model
        super(TimmModelWithAttention, self).__init__()

        # CNN encoder from timm
        self.encoder = timm.create_model(
            backbone, in_chans=Config.in_chans, num_classes=Config.out_dim, pretrained=pretrained
            )
        
        hdim = self.encoder.num_features

        # Remove classifier heda
        self.encoder.classifier == nn.Identity()
        
        # Attention layer 1
        self.attention1 = nn.Sequential(
            nn.Linear(hdim, hdim // 2), nn.ReLu(), nn.Linear(hdim // 2, 1), nn.Sigmoid()
        )

        # Attention layer 2
        self.attention2 = nn.Sequential(
            nn.Linear(hdim, hdim // 2), nn.ReLu(), nn.Linear(hdim // 2, 1), nn.Sigmoid()
        )

        # Attention layer 3
        self.attention3 = nn.Sequential(
            nn.Linear(hdim, hdim // 2), nn.ReLu(), nn.Linear(hdim // 2, 1), nn.Sigmoid()
        )

        # LSTM (bidirect.)
        self.lstm = nn.LSTM(
            input_size=hdim, hidden_size=256, num_layers=2, dropout=0.0, bidirectional=True, batch_first=True
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(2*256, 256) # 2*256 because bidirectional
            , nn.BatchNorm1d(256), nn.Dropout(0.25), nn.LeakyReLu(0.1), nn.Linear(256, Config.out_dim)
        )


    def forward(self, x):
        """X: (batch_size, nperslice, C, H, W)"""

        bs = x.size(0)

        # Flatten the slices to single images
        x = x.view(bs * Config.n_per_slice, Config.in_chans,Config.image_size, Config.image_size)
        feat = self.encoder() # bs*nperslice, hdim

        attention_weights1 = self.attention1(feat)
        feat = feat* attention_weights1

        
        attention_weights2 = self.attention2(feat)
        feat = feat* attention_weights2

        
        attention_weights3 = self.attention3(feat)
        feat = feat* attention_weights3

        feat = feat.view(bs, Config.n_per_slice, -1) # reshape to batch, sequence, feature_dim for the LSTM
        feat, _ = self.lstm(feat)# output: (bs, n_perslice, 512)
        feat = feat[:, -1, :] # using last time step featres
        logits = self.head(feat)
        return logits
    

################################
# Inference of the dataset
################################

class Inference(Dataset):
    def __init__(self, df, transform):
        """
        Args:
            df (DataFrame): DataFrame with columns 'StudyInstanceUID' and 'c' (vertebra id)
            transform: Albumentations transform to apply on each slice.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
         return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        cid = row.c
        uid = row.StudyInstanceUID
        images = []

        for slice_idx in range(Config.n_per_slice):
            filepath = os.path.join(Config.data_dir, f'{uid}_{cid}_{slice_idx}.npy')
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            img = np.load(filepath)

            # Apply transform
            augmented = self.transform(image=img)
            image = augmented['image']
            
            # Transpose and normalize
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
            images.append(image)

        # Stack to shape: (n_per_slice, C, H, W)
        images = np.stack(images, axis=0)
        return torch.tensor(images).float(), uid, cid
 



