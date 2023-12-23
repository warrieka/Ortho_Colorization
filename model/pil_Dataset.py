from PIL import Image
import numpy as np
from skimage.color import rgb2lab
from skimage.exposure import adjust_gamma
from skimage.util import random_noise
from skimage.filters import gaussian
import torch, os
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2
from typing import Iterable

def makeSubsetsFromList(list_imgs:Iterable[os.PathLike], 
                        size:int=4000, test_size:int=500, all:bool=False):
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

class ColorizationDataset(Dataset):
    def __init__(self, paths:Iterable[os.PathLike], imsize:int=256, rootDir:str='', resize:bool=True):
        super().__init__()
        self.size = imsize
        self.paths = paths
        self.root = rootDir
        self.resize = resize

    def _aug(self, img:np.ndarray):
        "some data augmentation on img "
        img = adjust_gamma(img, gamma=np.random.uniform(low=0.5, high=1.5) ) #change lighting
        if np.random.uniform() > 0.8:  #add random noise 20% of the time
            img = (random_noise(img)*255).astype('uint8') 
        if np.random.uniform() > 0.9:  #add some random blur 10% of the time
            img = (gaussian(img, np.random.uniform(0,1.5) )*255).astype('uint8') 
        return img
    
    def __getitem__(self, idx:int):
        img = Image.open( Path(self.root) / self.paths[idx] ).convert("RGB")
        if self.resize or img.size[0] < self.size:
            new_size = img.shape[0] // self.resize
            img = img.resize( (new_size ,new_size), resample=Image.Resampling.LANCZOS )
        else:
            xpad = (img.size[0] - self.size)//2
            ypad = (img.size[1] - self.size )//2
            img = img.crop((xpad, ypad, self.size+xpad, self.size+ypad))

        img_lab = rgb2lab( self._aug( np.array(img) )) # Converting RGB to L*a*b

        img_lab = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)
    
