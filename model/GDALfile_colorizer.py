from osgeo import gdal #> https://pcjericks.github.io/py-gdalogr-cookbook/
gdal.UseExceptions() 
from pathlib import Path
import numpy as np, os, torch
from .mainModel import MainModel
from .tools import lab_to_rgb
from typing import Iterable
from skimage.exposure import match_histograms, adjust_sigmoid
from skimage import io

class GDALtile():
    def __init__(self, in_array:np.array, posx:int, poxy:int, padx:int, pady:int):
        self._Array = in_array
        self.posx = posx
        self.posy = poxy
        self.padx = padx
        self.pady = pady

    def setArray(self, resultArray):
        self._Array = resultArray

    def getArray(self) -> np.array:
        return self._Array 
    

class GDALfile_colorizer():
    def __init__(self, in_GDALfile:os.PathLike, weigths:os.PathLike, 
                 tileSize:int=512, refImage:os.PathLike=None, improve_contrast:bool=False):
        self.inDataset = gdal.Open( str(in_GDALfile), gdal.GA_ReadOnly)
        self.transform = np.array( self.inDataset.GetGeoTransform() )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(weigths)
        self.xsize = self.inDataset.RasterXSize
        self.ysize = self.inDataset.RasterYSize
        self.tileSize = tileSize
        self.refImage = None 
        self.improve_contrast = improve_contrast
        if refImage is not None:
            imh_h = gdal.Open( str(refImage), gdal.GA_ReadOnly) if refImage else None 
            self.refImage = imh_h.ReadAsArray()[:3]
        
    def _load_model(self, weigths):
        _model = MainModel()
        _model.eval()
        with torch.inference_mode():
            _model.load_state_dict(
                torch.load(weigths, map_location=self.device) )
        return _model
            
    def _infer(self, tile):
        T_tile = torch.Tensor( ( tile.getArray() /128) -1 ).unsqueeze(0)
        with torch.inference_mode():
            pred = self.model.net_G( T_tile.unsqueeze(0).to(self.device) )
        C_tile = lab_to_rgb(T_tile.unsqueeze(0), pred.cpu())[0]
        tile.setArray( (C_tile * 255).astype(np.uint8) )
        return tile

    def _balansColor(self, img):
        if self.refImage is None:
            return img
        return match_histograms(img , self.refImage, channel_axis=2 )

    def _process_tiles(self ):
        full_img= np.zeros( (3, self.ysize ,self.xsize) , dtype=np.uint8  ) 
        for tile in self._tileGenerator(self.tileSize):
            newTile = self._infer(tile)

            xoff = newTile.posx
            yoff = newTile.posy
            patch = adjust_sigmoid( newTile.getArray() ) if self.improve_contrast else newTile.getArray()

            if newTile.pady > 0:
                patch = patch[0:-1 *tile.pady, :]
            if newTile.padx > 0:
                patch = patch[:, 0:-1*tile.padx]

            bil = np.stack((patch[:,:,0], patch[:,:,1], patch[:,:,2]), axis=0).astype(np.uint8)
            full_img[:, yoff: yoff + bil.shape[1] , xoff: xoff + bil.shape[2] ] = bil

        return full_img

    def _tileGenerator(self, imsize):
        for xi, xoff in enumerate( range(0, self.xsize , imsize)):
            for yi, yoff in  enumerate( range(0, self.ysize , imsize)):
                # Last row and column are smaller then needed for inference                
                xsize, ysize = (imsize,imsize)
                if (self.xsize - (xi*imsize)) < imsize:
                    xsize =  self.xsize - (xi*imsize)
                if (self.ysize - (yi*imsize)) < imsize:
                    ysize =  self.ysize - (yi*imsize)

                img = self.inDataset.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize)
                img = np.pad(img, ((0, imsize- ysize),(0, imsize- xsize)),  mode='median') 

                yield GDALtile(img, xoff, yoff, imsize- xsize , imsize- ysize)

    def saveOutDataSet(self, out_GDALfile:os.PathLike, outDriver:str='GTiff', creation_options:Iterable=None):
        img_rgb = self._process_tiles() #[:,:self.ysize,:self.xsize]
        if self.refImage is not None:
            img_rgb = np.stack(
                [ match_histograms(img_rgb[c] , self.refImage[c]) for c in range(3) ]
            , axis=0)

        if outDriver in ('PNG', 'JPEG', 'JPEG2000') or outDriver is None:
            fname = Path(out_GDALfile)
            wld_ext = '.wld' if outDriver != 'JPEG2000' else '.j2w'
            io.imsave( fname=fname, arr= img_rgb.transpose((1, 2, 0)) )
            with open( fname.parent / f"{fname.stem}{wld_ext}" ,'w') as wld:
                wld.write("\n".join([str(self.transform[i]) for i in (1,2,4,5,0,3)]))
            return 

        drv = gdal.GetDriverByName(outDriver)
        if creation_options is None and outDriver == 'GTiff':
            creation_options = [ 
                'BIGTIFF=IF_NEEDED', 'INTERLEAVE=BAND', 'COMPRESS=JPEG',
                'Tiled=YES', f'BLOCKSIZE={self.tileSize}', 
                'SPARSE_OK=True', 'NUM_THREADS=ALL_CPUS' ]

        self.outDataset = drv.Create(str(out_GDALfile), bands=3, xsize=self.xsize, ysize=self.ysize,
                                                eType=gdal.GDT_Byte, options=creation_options) 
        self.outDataset.WriteArray(img_rgb)
        self.outDataset.SetGeoTransform(self.transform)
        self.outDataset.SetProjection(self.inDataset.GetProjection())
        # free up memory
        self.outDataset.FlushCache()
        self.outDataset = None
        self.inDataset = None