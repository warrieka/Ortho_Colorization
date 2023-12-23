from osgeo import gdal
gdal.UseExceptions()
import torch
import numpy as np
from skimage.color import rgb2lab
from skimage.exposure import adjust_gamma
from skimage.util import random_noise
from skimage.filters import gaussian
from torch.utils.data import IterableDataset
from torchvision.transforms import v2

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
        self.itr = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=False)]) #Convert to ImageTensor
    
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


