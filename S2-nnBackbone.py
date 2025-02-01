# Program: Cervical fracture detection of c1-c7, stage 2
# Author: Mec Glandorff
# Version: 3.0
# Description: This program  uses a neural network to detect cervical fractures in the c1-c7 region from CT data. The CT data  
#              is first segmented in 15 6x224x224 numpy matrices (see more details in configuration class) per vertebra per patient. 
#              It combines a backbone CNN (or later Vision Transformer) for feature extraction with a LSTM for sequence modeling. 
#              This gives the benefit that it can detect anomolies/fractures across ct slices. It does so per vertebrae.
#              I chose to run this on my own pc (GPU: RTX 4060) instead of in colab pro+, which doesn't make sense 
#              from a computational perspective, but it seemed interesting to run it locally. Which gave a whole lot of new 
#              issues to deal with. Mostly CUDA issues, which were fairly easy to resolve and it turns out you can 
#              crash a pc so hard it fully removes its GPU driver making you think you might have ruined your GPU :3 for a few restarts. 



import os
import gc
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import timm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


########################################3
# Configuration Class
#########################################
class Config:
    data_dir = "data/masks/"
    csv_path = "data/masks/train_seg.csv"
    
    # Image and model Parameters              #### So the actual CT-data has image size 512x512, but has been downsized to 224x224 because of computational limits.
    image_size = 224                          #### Upsizing this to 512x512 would likely improve performance, certainly if accompanied by additional CT-data (patients).
    n_slice_per_c = 15                        #### 15 slices per vertebra.
    in_chans = 6                              #### Input size is a numpy matrix with size 6x224x224. 5 channels are 8bit grayscale slices of vertebra, 6th slice a vertebra mask
    out_dim = 1  # Binary classification!
    
    # Training Parameters
    batch_size = 6                            #### I think that a more optimal batch size would be between 8-32 and then you could also leave out accumulation steps. 
    accumulation_steps = 3                    #### Again computational limits, so simulated batch size 24 by adding 6 accumulation steps for batch_size 4. 
    init_lr = 1e-4
    eta_min = 1e-6
    n_epochs = 25                             #### The few runs I did with efficientnetv2 medium and small usually converges around 7-15 epochs.
    num_workers = 4                           #### 4 num_workers gives a slight CPU bottleneck, which was kind of convenient to decrease the temperature of my workstation.
    use_amp = True                            #### Wouldn't have used AMP if I didn't have computational limits. Don't think it impacts accuracy that much in this case.
    drop_rate = 0.0                           #### Increase next run, see if I can get further than 12 epochs before convergence. 
    drop_rate_last = 0.3
    backbone = 'tf_efficientnetv2_s_in21ft1k' ### Switch this with other model later (swin transformer, however then architecture should be adjusted slightly).
                                              ### For now I used EfficientNetV2, I used the small and medium variants (small variant being sort of fast for training).
                                              ### I used the in21ft1k pretained version, which was pretrained on imagenet-21k and finetuned on 1k. 
                                              ### I think this backbone is quite good, but didn't put much time/research in it as my main priority was finding out 
                                              ### whether this architecture would work and checking with augmentations.
                                              ### Keep in mind that the augmentation can impact other models differently.
    
    # Reproducibility seed stuff
    seed = 42
    
    # Model + logging
    model_dir = './models'
    log_dir = './logs'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Early Stopping
    early_stopping_patience = 5  # Number of epochs to wait for improvement focusses on max recall for now. 

    # Example Vertebra Weights (customize for testing?)
    # If you want all vertebrae equally weighted, set all=1.0
    # If C1 is more critical or rarer, might set 2.0, etc.

    # For now what I did was take the % of label 1 per vertebra ['7.23', '14.12', '3.62', '5.35', '8.03', '13.73', '19.47']% for c1-c7 respectively 
    # and 47.62% (sum of all %) for patient overall. I then inverted the normalized frequencies as weights with the smallest weight (most common fracture (c7)) as 1.0. 
    vertebra_weights = {1: 2.69, 2: 1.38, 3: 5.38, 4: 3.64, 5: 2.43, 6: 1.42, 7: 1.0}


###########################################
# Focal Loss Implementation
###########################################
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='none'):
        """
        Focal Loss for binary classification
        Args:
            alpha (float): Weighting factor for the positive class
            gamma (float): Focusing parameter to down-weight easy examples
            reduction (str): 'none' to apply sample-wise weighting externally
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')  
    
    def forward(self, logits, targets):
        """
        Forward pass
        Args:
            logits (Tensor): Shape (batch_size,)
            targets (Tensor): Shape (batch_size,)
        """
        # Compute BCE loss
        bce_loss = self.bce_with_logits(logits, targets)
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Compute pt
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # Compute alpha_t
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal term
        focal_term = alpha_t * (1 - pt) ** self.gamma
        
        # Compute final loss
        loss = focal_term * bce_loss
    
        return loss


########################
# Dataset Class
########################
class CLSDataset(Dataset):
    def __init__(self, df, mode, transform):
        """
        Custom Dataset for the vertebra classification
        Args:
            df (DataFrame): DataFrame containing data info (StudyInstanceUID, c, label)
            mode (str): 'train' or 'valid'
            transform (albumentations.Compose): Transformations to apply on the data
        """

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        cid = row.c  # Vertebra ID (1-7), I named it cid so c1,c2,c3,etc is what output looks like
        uid = row.StudyInstanceUID # Patient ID trackers
        images = [] # Actual matrices of images

        for slice_idx in range(Config.n_slice_per_c):
            filepath = os.path.join(Config.data_dir, f'{uid}_{cid}_{slice_idx}.npy')
            img = np.load(filepath)

            # Albumentations expects a dict with key='image' by default, so I had to add this
            augmented = self.transform(image=img)
            image = augmented['image']

            # Convert shape (H, W, C) -> (C, H, W) and scale
            image = image.transpose(2, 0, 1).astype(np.float32) / 255.
            images.append(image)

        # Shape: (n_slice_per_c, C, H, W)
        images = np.stack(images, axis=0)

        # Build item
        if self.mode in ['train', 'valid']:
            label = torch.tensor(row.label, dtype=torch.float32)

            # Retrieve vertebra-specific weight from config
            vertebra_weight = Config.vertebra_weights[int(cid)]
            return (
                torch.tensor(images).float(),  # Vertebra tensor, shape (15, C, H, W)
                label,
                torch.tensor(cid, dtype=torch.long),
                torch.tensor(vertebra_weight, dtype=torch.float32)
            )
        else:
            raise ValueError("Error")

##############################################
# Model Definition with Attention
##############################################
class TimmModelWithAttention(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmModelWithAttention, self).__init__()

        # Init CNN encoder
        self.encoder = timm.create_model(
            backbone,
            in_chans=Config.in_chans,
            num_classes=Config.out_dim,
            pretrained=pretrained
        )
        hdim = self.encoder.num_features  

        # Remove the classifier head
        self.encoder.classifier = nn.Identity()

        # Attention layer 1
        self.attention1 = nn.Sequential(
            nn.Linear(hdim, hdim // 2),
            nn.ReLU(),
            nn.Linear(hdim // 2, 1),
            nn.Sigmoid()
        )

        # Attention layer 2
        self.attention2 = nn.Sequential(
            nn.Linear(hdim, hdim//2),
            nn.ReLU(), 
            nn.Linear(hdim//2, 1),
            nn.Sigmoid()
        )

        # Attention layer 3
        self.attention3 = nn.Sequential(
            nn.Linear(hdim, hdim//2),
            nn.ReLU(),
            nn.Linear(hdim//2, 1),
            nn.Sigmoid()
        )

        # LSTM layer (bidirectional)
        self.lstm = nn.LSTM(
            input_size=hdim,
            hidden_size=256,    #### Layers & hidden size can be increased to 512 / 3 or 4 if you have the computational power, 
            num_layers=2,       #### more data and (optionally) use a pretaraind transformer as backbone.
                                #### This will likely increase the model metrics.  
            dropout=Config.drop_rate,
            bidirectional=True, 
            batch_first=True
        )

        # Combine multiple layers in sequential pipeline
        self.head = nn.Sequential(
            nn.Linear(512, 256), # 512, because bidirectional! So 512-dimensional input vector and 256 dimensional output.
            nn.BatchNorm1d(256), 
            nn.Dropout(Config.drop_rate_last),
            nn.LeakyReLU(0.1),  # Small slope for negative values (0.1), neuron life support for dying neurons. 
            nn.Linear(256, Config.out_dim) # Final output dimension of 256 x 1. 
        )

    def forward(self, x):
        """
        x: (batch_size, n_slice_per_c, C, H, W)
        Where C = channels, H = height and W = width. 
        """
        bs = x.size(0)
        # Flatten slices into 2D
        x = x.view(bs * Config.n_slice_per_c, Config.in_chans, Config.image_size, Config.image_size) 
        feat = self.encoder(x)  # Extract high dim feature representation per vertebra (bs*n_slice_per_c, hdim)

        # Attention block 1
        att_weights1 = self.attention1(feat)  # compute attention weights (bs*n_slice_per_c, 1)
        feat = feat * att_weights1  # apply attention weights (bs*n_slice_per_c, hdim)

        # Attention block 2
        att_weights2 = self.attention2(feat)  # compute attention weights (bs*n_slice_per_c, 1)
        feat = feat * att_weights2  # apply attention weights (bs*n_slice_per_c, hdim)

        # Attention block 3
        att_weights3 = self.attention3(feat)  # compute attention weights (bs*n_slice_per_c, 1)
        feat = feat * att_weights3  # apply attention weights (bs*n_slice_per_c, hdim)



        # Reshape to LSTM input (3D:  batch, sequence, featuredim)
        feat = feat.view(bs, Config.n_slice_per_c, -1)  # (bs, n_slice_per_c, hdim)
        feat, _ = self.lstm(feat)  # (bs, n_slice_per_c, 512)

        # Take the last time-step of which only these features are retained.
        feat = feat[:, -1, :]  #Extract to (bs, 512)

        # Final features through classification head. 
        logits = self.head(feat)  # (bs, out_dim)
        return logits

############################
# Training Function #
############################
def train_func(model, loader_train, optimizer, scaler, focal_loss):
    """
    Training loop for one epoch
    Returns: (train_loss, train_auc, train_f1, train_recall, train_acc)
    """
    model.train()
    train_loss_list = []
    all_targets = []
    all_preds = []
    all_probs = []

    optimizer.zero_grad()
    bar = tqdm(loader_train, desc="Training", leave=True)

    for step, (images, targets, cids, vertebra_wgts) in enumerate(bar):
        images = images.cuda()
        targets = targets.cuda()
        vertebra_wgts = vertebra_wgts.cuda()  # shape (batch_size,)

        with torch.cuda.amp.autocast(enabled=Config.use_amp):
            logits = model(images).squeeze(-1)  # shape (batch_size,)
            # FocalLoss returns a per-sample loss vector
            losses = focal_loss(logits, targets)
            # Multiply by per-vertebra weight
            losses = losses * vertebra_wgts
            # Average the batch
            loss = losses.mean()  
            # Gradient accumulation
            loss = loss / Config.accumulation_steps

        scaler.scale(loss).backward()

        # Add accumulation steps to simulate larger batch size, since my VRAM is too small to handle 8-32+ batch size (which I would prefer).
        if (step + 1) % Config.accumulation_steps == 0:
            # Reverse scaler, because AMP scales it to avoid underflow, but we want the original scale for the next clipping step. 
            scaler.unscale_(optimizer)
            # Gradient clipping to avoid exploding gradients. This might be needing tweeking? I just coded it out without many thought of what max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Updating Gradscaler with gradients and parameters from previous steps, helps AMP maintain num stability during training.
            scaler.step(optimizer)
            scaler.update()
            # Clear accum. gradients for the next cycle.
            optimizer.zero_grad()

        # Keep track of the average (undo the division by accumulation_steps for logging)
        train_loss_list.append(loss.item() * Config.accumulation_steps)

        # Predictions
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        # Maybe in the future change 0.5 with a Config.Threshold value. Since we might want to adjust the prob threshold to balance Recall, F1, accuracy. 
        preds = (probs >= 0.5).astype(int)

        # Store probabilities, predictions and targets below
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

        # Update progress bar in terminal. 
        bar.set_postfix(loss=np.mean(train_loss_list))

    # Handle any leftover grads
    if (step + 1) % Config.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    # Metrics
    train_loss = np.mean(train_loss_list)
    try:
        train_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        train_auc = float('nan')
    train_f1 = f1_score(all_targets, all_preds, zero_division=0)
    train_recall = recall_score(all_targets, all_preds, zero_division=0)
    train_acc = accuracy_score(all_targets, all_preds)

    return train_loss, train_auc, train_f1, train_recall, train_acc

#################################
# Validation Function
#################################
def valid_func(model, loader_valid, focal_loss):
    """
    Validation function.
    Returns: (val_loss, val_auc, val_f1, val_recall, val_acc, metrics_per_c)
    """

    # Disable dropout and batch normalization for validation
    model.eval()

    # Lists of values, this is per batch
    val_loss_list = []
    all_targets = []
    all_preds = []
    all_probs = []
    all_cids = []

    bar = tqdm(loader_valid, desc="Validating", leave=True)

    with torch.no_grad():
        for images, targets, cids, vertebra_wgts in bar:    #images: batch, target: True label, cids: vertebra c_id, wghts= weights.

            # Move everything to GPU. 
            images = images.cuda()
            targets = targets.cuda()
            vertebra_wgts = vertebra_wgts.cuda()

            # Pass forward 
            logits = model(images).squeeze(-1)  # squeeze, to remove last dimension --> (batch_size,)

            # Focall losses handling
            losses = focal_loss(logits, targets)
            losses = losses * vertebra_wgts
            val_loss_list.append(losses.mean().item())

            probs = torch.sigmoid(logits).detach().cpu().numpy() #Sigmoid to get probabilities in [0,1]
            preds = (probs >= 0.5).astype(int)                   ### Threshold, in the future I might check whether 0.5 is appropriate,
                                                                 ### as focus for this issue is recall not accuarcy! Maybe add threshold to config as suggested before?
            
            # Store the metrics retrieved
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())
            all_cids.extend(cids.detach().cpu().numpy())

            # Update progress bar
            bar.set_postfix(loss=np.mean(val_loss_list))

    val_loss = np.mean(val_loss_list)
    try:
        val_auc = roc_auc_score(all_targets, all_probs) # Area under curve
    except ValueError:
        val_auc = float('nan')
    val_f1 = f1_score(all_targets, all_preds, zero_division=0) # F1
    val_recall = recall_score(all_targets, all_preds, zero_division=0) # Recall
    val_acc = accuracy_score(all_targets, all_preds)    # Accuracy

    # Metrics per vertebra c
    metrics_per_c = {}
    unique_cs = np.unique(all_cids)
    for c in unique_cs:
        idx = np.where(np.array(all_cids) == c)
        c_targets = np.array(all_targets)[idx]
        c_preds = np.array(all_preds)[idx]
        c_probs = np.array(all_probs)[idx]

        c_f1 = f1_score(c_targets, c_preds, zero_division=0)
        c_recall = recall_score(c_targets, c_preds, zero_division=0)
        c_acc = accuracy_score(c_targets, c_preds)
        try:
            c_auc = roc_auc_score(c_targets, c_probs)
        except ValueError:
            c_auc = float('nan')

        metrics_per_c[int(c)] = {
            "f1": c_f1,
            "recall": c_recall,
            "accuracy": c_acc,
            "auc": c_auc
        }

    return val_loss, val_auc, val_f1, val_recall, val_acc, metrics_per_c

############################
# Main Training Loop
# -----------------------------
def run():
    # Set seeds
    torch.manual_seed(Config.seed)
    torch.cuda.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    random.seed(Config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    df = pd.read_csv(Config.csv_path)

    # Select patients_amount
    patient_ids = df['StudyInstanceUID'].unique()
    if len(patient_ids) < 100:
        raise ValueError("The dataset contains fewer than 100 patients!")
    patients_amount = 2018
    selected_patients = np.random.choice(patient_ids, size=patients_amount, replace=False)
    df = df[df['StudyInstanceUID'].isin(selected_patients)].reset_index(drop=True)
    print(f"Selected 100 patients for the trial run. Filtered dataset has {len(df)} rows.")

    # Expand DataFrame
    study_list, c_list, label_list = [], [], []
    for _, row in df.iterrows():
        for c in range(1, 8):
            study_list.append(row.StudyInstanceUID)
            c_list.append(c)
            label_list.append(row[f"C{c}"])

    expanded_df = pd.DataFrame({
        "StudyInstanceUID": study_list,
        "c": c_list,
        "label": label_list
    })

    print(f"Total expanded samples: {len(expanded_df)}")

    # Remove duplicates if any
    duplicates = expanded_df.duplicated().sum()
    print(f"Number of duplicate samples: {duplicates}")
    if duplicates > 0:
        expanded_df.drop_duplicates(inplace=True)
        expanded_df.reset_index(drop=True, inplace=True)
        print(f"Samples after removing duplicates: {len(expanded_df)}")

    # Class weights for positive/negative
    labels_array = expanded_df['label'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels_array),
        y=labels_array
    )
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
    print("Binary Class Weights:", class_weights_dict)

    # Focal loss
    focal_loss = FocalLoss(alpha=class_weights_dict[1], gamma=2, reduction='none').cuda()

    # Train-Val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=Config.seed)
    train_idx, val_idx = next(sss.split(expanded_df, expanded_df["label"]))

    train_df = expanded_df.iloc[train_idx].reset_index(drop=True)
    val_df = expanded_df.iloc[val_idx].reset_index(drop=True)

    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Augmentations
    transforms_train = albumentations.Compose([
        albumentations.Resize(Config.image_size, Config.image_size),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=0.1, contrast_limit=0.1, p=0.5
        ),
        albumentations.GaussNoise(var_limit=(2.0, 7.0), p=0.3),
        albumentations.Cutout(                                              # Maybe skip cutout???~~
            max_h_size=int(Config.image_size * 0.2),
            max_w_size=int(Config.image_size * 0.2),
            num_holes=2,
            p=0.3                                                           
        ),
        albumentations.Normalize(
            mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
        ),
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(Config.image_size, Config.image_size),
        albumentations.Normalize(
            mean=(0.485, 0.456, 0.406, 0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225, 0.229, 0.224, 0.225)
        ),
    ])

    # Datasets and loaders
    dataset_train = CLSDataset(train_df, mode='train', transform=transforms_train)
    dataset_val = CLSDataset(val_df, mode='valid', transform=transforms_val)

    loader_train = DataLoader(
        dataset_train,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        drop_last=True,
        pin_memory=True
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        drop_last=False,
        pin_memory=True
    )

    # Model
    model = TimmModelWithAttention(Config.backbone, pretrained=True).cuda()

    # Optimizer + learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.init_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True,
        min_lr=Config.eta_min
    )

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_amp)

    # Early stopping
    best_recall = 0.0
    epochs_no_improve = 0

    for epoch in range(1, Config.n_epochs + 1):
        print(f"\nEpoch [{epoch}/{Config.n_epochs}]")

        # ---- Training ----
        train_loss, train_auc, train_f1, train_recall, train_acc = train_func(
            model, loader_train, optimizer, scaler, focal_loss
        )

        # ---- Validation ----
        val_loss, val_auc, val_f1, val_recall, val_acc, metrics_per_c = valid_func(
            model, loader_val, focal_loss
        )

        # Prepare log strings
        train_msg = (f"Train Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | "
                     f"F1: {train_f1:.4f} | Recall: {train_recall:.4f} | Acc: {train_acc:.4f}")
        val_msg = (f"Val   Loss: {val_loss:.4f}   | AUC: {val_auc:.4f}   | "
                   f"F1: {val_f1:.4f}   | Recall: {val_recall:.4f}   | Acc: {val_acc:.4f}")

        # Print to console
        print(train_msg)
        print(val_msg)

        # Print per-vertebra stats
        per_vertebra_msg = "Val metrics per vertebra c:\n"
        for c_id, metrics in metrics_per_c.items():
            per_vertebra_msg += (f"  C{c_id}: F1={metrics['f1']:.4f}, "
                                 f"Recall={metrics['recall']:.4f}, "
                                 f"Acc={metrics['accuracy']:.4f}, "
                                 f"AUC={metrics['auc']:.4f}\n")
        print(per_vertebra_msg)

        # ---- Save log to file ----
        log_filename = os.path.join(Config.log_dir, f"loginfo_epoch{epoch}.txt")
        with open(log_filename, "w") as f:
            f.write(f"Epoch [{epoch}/{Config.n_epochs}]\n")
            f.write(train_msg + "\n")
            f.write(val_msg + "\n")
            f.write(per_vertebra_msg + "\n")

        # Scheduler steps on validation recall
        scheduler.step(val_recall)

        # Early stopping check
        if val_recall > best_recall:
            best_recall = val_recall
            epochs_no_improve = 0

            # Save best model
            best_model_path = os.path.join(Config.model_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  [*] Best model updated. Recall={best_recall:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement in recall for {epochs_no_improve} epoch(s).")

        # ---- Save model for this epoch ----
        latest_model_path = os.path.join(Config.model_dir, f"latest_model_epoch{epoch}.pth")
        torch.save(model.state_dict(), latest_model_path)
        print("  Latest model saved.")

        if epochs_no_improve >= Config.early_stopping_patience:
            print(f"Early stopping triggered (no recall improvement for {epochs_no_improve} epochs).")
            break

    print("\nTraining complete.")
    del model
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    run()
