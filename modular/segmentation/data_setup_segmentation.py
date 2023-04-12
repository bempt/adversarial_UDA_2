"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import zipfile



## Split data
def create_df(image_path):
    name = []
    for dirname, _, filenames in os.walk(image_path):
        for filename in filenames:
            name.append(filename.split('.')[0])

    df = pd.DataFrame({'id': name}, index = np.arange(0, len(name)))
    
    img_count = len(df)
    
    return df, img_count

def split_data_supervised(df, val_split, test_split, random_state):
    X_trainval, X_test = train_test_split(df['id'].values, test_size=test_split, random_state=random_state)
    X_train, X_val = train_test_split(X_trainval, test_size=val_split, random_state=random_state)
    return X_trainval, X_test, X_train, X_val

def split_data_unsupervised(df, val_split, random_state):
    X_train, X_val = train_test_split(df['id'].values, test_size=val_split, random_state=random_state)
    return X_train, X_val

## Verify data
def data_size(img_count, X_train, X_val, X_test):
    print('Total images: ', img_count)
    print('Train Size   : ', len(X_train))
    print('Val Size     : ', len(X_val))
    print('Test Size    : ', len(X_test))

def mask_over_image(df, file_id, image_path, mask_path, alpha=0.6):
    img = Image.open(image_path + df['id'][file_id] + '.jpg')
    mask = Image.open(mask_path + df['id'][file_id] + '.png')
    print('Image Size', np.asarray(img).shape)
    print('Mask Size', np.asarray(mask).shape)
    plt.imshow(img)
    plt.imshow(mask, alpha=alpha)
    plt.title('Sample Image with Mask Applied')
    plt.show()

'''
Semantic-drone
'''
import os
import pandas as pd
import numpy as np
import json
import cv2
from PIL import Image
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch


N_CLASSES = 23


# Train and val dataset
class SemanticDroneDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)
        mask = torch.from_numpy(mask).long()
        
        if self.patches:
            img, mask = self.tiles(img, mask)
            
        return img, mask
    
    def tiles(self, img, mask):

        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768) 
        img_patches  = img_patches.contiguous().view(3,-1, 512, 768) 
        img_patches = img_patches.permute(1,0,2,3)
        
        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)
        
        return img_patches, mask_patches
    

## Train and val augmentations
t_train = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
                     A.HorizontalFlip(),
                     A.VerticalFlip(), 
                     A.GridDistortion(p=0.2),
                     A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                     A.GaussNoise()
                     ])

t_val = A.Compose([A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
                   A.HorizontalFlip(),
                   A.GridDistortion(p=0.2)
                   ])


## Test dataset
class SemanticDroneTestDataset(Dataset):
    
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
      
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + '.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + '.png', cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            mask = aug['mask']
        
        if self.transform is None:
            img = Image.fromarray(img)
        
        mask = torch.from_numpy(mask).long()
        
        return img, mask

## Test augmentations
t_test = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)


from torch.utils.data import Dataset, DataLoader
from modular.segmentation import data_setup_segmentation as seg_data
import os

def make_datasets(image_path, mask_path,
                  train_split, val_split, test_split,
                  mean, std,
                  train_augmentation, val_augmentation, test_augmentation,
                  patch=False,
                  ):
    train_set = SemanticDroneDataset(image_path, mask_path,
                                     train_split,
                                     mean, std,
                                     train_augmentation, patch=False,
                                    )
    val_set = SemanticDroneDataset(image_path, mask_path,
                                      val_split,
                                      mean, std,
                                      val_augmentation, patch=False,
                                    )
    test_set = SemanticDroneTestDataset(image_path, mask_path,
                                        test_split,
                                        test_augmentation,
                                        )
    return train_set, val_set, test_set

def make_dataloaders(train_set, val_set, test_set,
                     batch_size,
                     num_workers_ratio,
                     shuffle=True,
                     ):
    
    num_workers = int(os.cpu_count() * num_workers_ratio)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False
                            )
    
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=False
                            )
    
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=False
                            )
    
    return train_loader, val_loader, test_loader