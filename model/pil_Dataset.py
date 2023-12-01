from PIL import Image
import numpy as np
from skimage.color import rgb2lab
from skimage.exposure import adjust_gamma
from skimage.filters import gaussian
import torch
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import v2

def makeSubsetsFromList(list_imgs, size=4000, test_size=500, all=False):
    """Make a subset of <jpgs> of <size> and reserve <test_size> for testing
    If <all> is true, the all images are used and <size> is ignored
    Returns randomised 2 lists: a Train list and a Test list containing the paths to the oimages
    """
    if all: size = len(list_imgs)
    #randomize without replacement, can pick same image only once 
    paths_subset = np.random.choice(list_imgs, size, replace=False) 
    rand_idxs = np.random.permutation(size) 
    train_idxs = rand_idxs[:size - test_size] 
    val_idxs = rand_idxs[size - test_size:] 
    train_paths = paths_subset[train_idxs] 
    test_paths = paths_subset[val_idxs]
    return train_paths, test_paths

def makeWeightedDatasetFromFeather(arrow, train_size=4500, test_size=500,
                                pathField='NAME', weigthField='WEIGHT', replacement=False):
    ds= pd.read_feather(arrow)
    train_paths = ds.sample(train_size, weights=weigthField, replace=replacement)[pathField]
    test_paths = ds.sample(test_size, weights=weigthField, replace=replacement)[pathField]
    return list(train_paths), list(test_paths)

class ColorizationDataset(Dataset):
    def __init__(self, paths, imsize=256, rootDir=''):
        super(ColorizationDataset).__init__()
        self.size = imsize
        self.paths = paths
        self.root = rootDir
        # self.aug = v2.RandomPhotometricDistort()

    def aug(self, img):
        "some simple data augmentation on img "
        return adjust_gamma( gamma= np.random.uniform(low=0.5, high=1.5), 
                             image= gaussian(image=img, sigma=(np.random.randint(0,4) , np.random.randint(0,4)) )  )
    
    def __getitem__(self, idx):
        img = Image.open( Path(self.root) / self.paths[idx]).convert("RGB").resize((self.size ,self.size ))
        img_lab = rgb2lab( self.aug( np.array(img) )) # Converting RGB to L*a*b

        img_lab = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])(img_lab)
        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)
    
