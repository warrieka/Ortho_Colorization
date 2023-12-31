from torchvision.io import read_image, ImageReadMode
import numpy as np
from skimage.color import rgb2lab
from skimage.util import random_noise
from skimage.transform import resize
import torch, os
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms.v2 import RandomResizedCrop, CenterCrop, Compose, GaussianBlur, ToDtype
from typing import Iterable

def makeSubsetsFromList(list_imgs:Iterable[os.PathLike], 
                        size:int=4000, test_size:int=400, all:bool=False):
    """Make a subset of <jpgs> of <size> and reserve <test_size> for testing
    If <all> is true, the all images are used and <size> is ignored
    Returns randomised 2 lists: a Train list and a Test list containing the paths to the oimages
    """
    if all: size = len(list_imgs)
    #randomize without replacement, can pick same image only once 
    paths_subset = np.random.choice(list_imgs, size, replace=False) 
    rand_idxs   = np.random.permutation(size) 
    train_idxs  = rand_idxs[:size - test_size] 
    val_idxs    = rand_idxs[size - test_size:] 
    train_paths = paths_subset[train_idxs] 
    test_paths  = paths_subset[val_idxs]
    return train_paths, test_paths

def makeWeightedDatasetFromFeather(arrow:os.PathLike, size:int=4000,
             pathField:str='path', weigthField:str='WEIGHT', replacement:bool=False):
    ds= pd.read_feather(arrow)
    paths = ds.sample(size, weights=weigthField, replace=replacement)[pathField]
    return list(paths)

def grainify(img:np.ndarray):
    c, rows, cols = img.shape
    val = np.random.uniform(0.036, 0.107)**2 

    # Full resolution
    noise_1 = np.zeros((rows, cols))
    noise_1 = random_noise(noise_1, mode='gaussian', var=val, clip=False)

    # # Half resolution
    noise_2 = np.zeros((rows//2, cols//2))
    noise_2 = random_noise(noise_2, mode='gaussian', var=(val*2)**2, clip=False)  
    noise_2 = resize(noise_2, (rows, cols))  # Upscale to original image size

    noise = noise_1 + noise_2 
    noise = np.stack( [noise]*c, axis=0)
    
    noisy_img = img/255 + noise # Add noise_im to the input image.
    return np.round((255 * noisy_img)).clip(0, 255).astype(np.uint8)

class ColorizationDataset(Dataset):
    def __init__(self, paths:Iterable[os.PathLike], imsize:int=256, rootDir:str='', resize:bool=True):
        super().__init__()
        self.size = imsize
        self.paths = paths
        self.root = rootDir
        self.resize = resize

    def _aug(self, img):
        "some data augmentation on img "
        if self.resize:
            minscale = self.size / img.size(1)
            img = RandomResizedCrop( self.size, (minscale,1), antialias=True)(img)
        else:
            img = CenterCrop(self.size)(img)
        
        if np.random.uniform() > 0.7:
           img = torch.tensor( grainify(img.numpy()) )
        return img

    def __getitem__(self, idx:int):
        img = read_image( str(Path(self.root) / self.paths[idx]) , ImageReadMode.RGB ) 
        img = self._aug(img)                              
        img_lab = rgb2lab( img.numpy() , channel_axis=0 ) # Converting RGB to L*a*b
        img_lab = torch.tensor(img_lab, dtype=torch.float16)
       
        L =  (img_lab[0] / 50. - 1).unsqueeze(0) # Between -1 and 1
        ab = img_lab[1:3] / 110  # Between -1 and 1
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)
    
