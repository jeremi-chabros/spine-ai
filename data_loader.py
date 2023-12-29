# -*- coding: utf-8

import os
import pandas as pd
from PIL import Image
import nibabel as nib
import numpy as np
import torch
from torch.utils import data

class MinMaxNormalize(object):
    "Normalizing the ndarray between zero and one using its minimum and maximum value"
    def __call__(self,sample):
        x = sample
        xmin = x.min()
        xmax = x.max()
        x = x-xmin
        if (xmax-xmin) != 0:
            x = x/(xmax-xmin)
        
        return x

class CenterCrop(object):
    """Crops the given np.ndarray at the center to have a region of
    """
    def __call__(self, sample):
        w, h = sample.shape[1], sample.shape[0]
        th, tw = self.finesize
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return sample[y1:y1+th, x1:x1+tw, :]

class convert(object):
    "convert to Tensor from numpy"
    def __call__(self,sample):
         x = sample
#         x = x.transpose(1,0)
         x = x.transpose(2,0,1).astype(np.float32)#converting from HXWXC to CXHXW
         return torch.from_numpy(x)
     

class ScanDataset(data.Dataset):
    """Scan dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        #self.annotations = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.png')
        image = Image.open(img_name)

        # Convert the PIL image to a numpy array and normalize
        image_np = np.array(image)
        xmin = image_np.min()
        xmax = image_np.max()
        image_np = (image_np - xmin) / (xmax - xmin)

        # If the image is grayscale (i.e., has only one channel), replicate it three times to get an RGB-like image
        if len(image_np.shape) == 2 or image_np.shape[2] == 1:
            image_np = np.stack((image_np,)*3, axis=-1)

        # Convert the RGB-like image back to a PIL image for further processing or visualization
        image = Image.fromarray((image_np * 255).astype(np.uint8), 'RGB')

        annotations = np.array(self.annotations.iloc[idx, 1])
        
        # annotations = np.array(binary_to_distribution(self.annotations.iloc[idx, 1])) # changed this
        
        if annotations == True:
            annotations = np.append(annotations,[0.0])
        else:
            annotations = np.append(annotations,[1.0])
        annotations = annotations.astype('float32').reshape(-1, 1)

        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

def binary_to_distribution(label):
    # If label is 1
    if label == 1:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    # If label is 0
    else:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]