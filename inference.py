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