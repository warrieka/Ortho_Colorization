from osgeo import gdal
gdal.UseExceptions()
import torch, os
from pathlib import Path
import pandas as pd
import numpy as np
from skimage.color import rgb2lab
from skimage.util import random_noise
from torch.utils.data import IterableDataset
from torchvision.transforms import v2, InterpolationMode
from torchvision.io import read_image, ImageReadMode
from .tools import grainify

class gdalTrainDataset(IterableDataset):
    def __init__(self, path, imsize=256, xSkip=0, yskip=0):
        super(gdalTrainDataset).__init__()
        self.size = imsize
        self.path = path
        self.skip = (xSkip,yskip)
        self.itr = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)])  #Convert to ImageTensor of float32
       
    def _aug(self, img:np.ndarray):
        "some data augmentation on img "
        #img = adjust_gamma(img, gamma=np.random.uniform(low=0.5, high=1.5) ) #change lighting
        if np.random.uniform() > 0.8:  #add random noise 20% of the time
            img = (random_noise(img)*255).astype('uint8') 
        # if np.random.uniform() > 0.7:  #add some random blur 30% of the time
        #     img = gaussian(img, np.random.uniform(0,1.5) )
        return img
    

    def __iter__(self):
        return iter(self.rasterGenerator(self.path, self.size, *self.skip))

    def rasterGenerator(self, ds_path, imsize=256, xSkip=0, yskip=0):
        ds = gdal.Open(ds_path,  gdal.GA_ReadOnly)
        workInfo = torch.utils.data.get_worker_info()
        xsize, ysize = (ds.RasterXSize, ds.RasterYSize)
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        if workInfo is None:
            xstart, ystart = (0,0)
        else:
            worker_id = workInfo.id
            xSize_worker = (xsize//imsize) 
            xstart = xSize_worker * worker_id
            ystart = 0

        for xoff in range(xstart, xsize-imsize, imsize + (imsize*xSkip) ):
            for yoff in range(ystart, ysize-imsize, imsize + (imsize*yskip) ):
                img = ds.ReadAsArray(xoff=xoff, yoff=yoff, xsize=imsize, ysize=imsize)

                if img  is None: 
                    break
                if np.median(img) in (0, 255, nodata): #skip mostly noddata, black or white tiles
                    continue
                
                img_aug = self.aug(  img.transpose((1, 2, 0))  )
                img_lab = self.itr( rgb2lab( img_aug ) ) #to Lab and ImageTensor

                L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
                ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
                yield {'L': L, 'ab': ab}

class gdalTestDataset(IterableDataset):
    def __init__(self, path, imsize=256):
        super(gdalTrainDataset).__init__()
        self.size = imsize
        self.path = path 
        self.itr = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]) #Convert to ImageTensor torch.bfloat16
    
    def __iter__(self):
        return self.rasterGenerator(self.path, self.size)

    def rasterGenerator(self, ds_path, imsize = 256):
        ds = gdal.Open(ds_path, gdal.GA_ReadOnly)

        xsize, ysize = (ds.RasterXSize, ds.RasterYSize)
        workInfo = torch.utils.data.get_worker_info()
        if workInfo is None:
            xstart, ystart = (0,0)
        else:
            worker_id = workInfo.id
            xSize_worker = (xsize//imsize) 
            if xSize_worker == 0:
                raise Exception('To many workers for this image, use less workers or use a larger image')
            xstart = xSize_worker * worker_id *imsize
            ystart = 0

        transformMatrix = np.array(ds.GetGeoTransform()) # Transformatation matrix pixel -> geo-coördinates

        for xoff in range(xstart, xsize -imsize, imsize):
            transformMatrix[0] = transformMatrix[0] + (imsize* transformMatrix[1])
            for yoff in range(ystart, ysize -imsize, imsize):
                transformMatrix[3] = transformMatrix[3] + (imsize* transformMatrix[5]) 
                img = ds.ReadAsArray(xoff=xoff, yoff=yoff, xsize=imsize, ysize=imsize)

                if img  is None:  
                    break
                
                x_img, y_img = img.shape[:2]
                if x_img != imsize or y_img != imsize:
                    np.pad(img, ((0,imsize- x_img),(0,imsize - y_img)))                      

                img_lab = self.itr( rgb2lab( img.transpose((1, 2, 0)) ) )
                L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
                ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1
                yield {'L': L, 'ab': ab, 'transform': transformMatrix}


class arrowDataset(IterableDataset):
    def __init__(self, arrow:os.PathLike, imsize:int=256, pathField:str='path', weightField:str='WEIGHT',
                 rootDir:os.PathLike='', grainify:bool=False, count:int=None ):
        super().__init__()
        self.size = imsize
        self.weightField = weightField
        self.pathField = pathField
        self.ds= pd.read_feather(arrow)
        self.c = count if count else len(self.ds)
        self.replace =  self.c > len(self.ds)//20
        self.root = Path( rootDir )
        
        augFunc = [           
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0, hue=0) , 
            v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.)), 
            v2.RandomResizedCrop(512, scale=(0.25, 1), interpolation= InterpolationMode.BICUBIC, antialias=None)  ]
        if grainify:
            augFunc.append(grainify)

        self.aug = v2.Compose(augFunc)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_num  = worker_info.num_workers if worker_info is not None else 1
        for _ in range(self.c // worker_num):
            path = self.ds.sample(weights=self.weightField, replace=self.replace)[self.pathField].item()
            img = read_image( str( self.root / path ) , ImageReadMode.RGB ) 
            img = self.aug(img)                              
            img_lab = rgb2lab( img.numpy() , channel_axis=0 ) # Converting RGB to L*a*b
            img_lab = torch.tensor(img_lab, dtype=torch.float32)
            L =  (img_lab[0] / 50. - 1).unsqueeze(0) # Between -1 and 1
            ab = img_lab[1:3] / 110  # Between -1 and 1
            yield {'L': L, 'ab': ab}
        
    def __len__(self):
        return self.c