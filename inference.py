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




