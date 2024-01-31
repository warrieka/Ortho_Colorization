from osgeo import gdal #> https://pcjericks.github.io/py-gdalogr-cookbook/
import warnings
from pathlib import Path
import numpy as np, os, torch
from .mainModel import MainModel
from .tools import lab_to_rgb
from typing import Iterable
from model.unet import ResUnet
from accelerate import Accelerator
from skimage import io

gdal.UseExceptions() 
warnings.simplefilter("ignore", category=RuntimeWarning)
warnings.simplefilter('ignore', category=UserWarning)

class GDALtile():
    def __init__(self, in_array:np.ndarray,posx:int, poxy:int, padx:int, pady:int, mask:np.ndarray=None ):
        self._Array = in_array
        self.mask = mask
        self.posx = posx
        self.posy = poxy
        self.padx = padx
        self.pady = pady

    def setArray(self, resultArray):
        self._Array = resultArray

    def getArray(self, nodata:int=None) -> np.array:
        array = self._Array 
        if nodata is not None and self.mask is not None:
            array[self.mask] = nodata
        return array
    

class GDALfile_colorizer():
    def __init__(self, in_GDALfile:os.PathLike, weigths:os.PathLike, tileSize:int=512, batch_size:int=32, unet_arch:str='resnet18'):
        print('Reading: ' ,in_GDALfile)
        self.inDataset = gdal.Open( str(in_GDALfile), gdal.GA_ReadOnly)
        self.transform = np.array( self.inDataset.GetGeoTransform() )
        self.accelerator = Accelerator(mixed_precision='fp16')
        self.device = self.accelerator.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = ResUnet(timm_model_name=unet_arch)
        self.model = self._load_model(weigths)
        self.xsize = self.inDataset.RasterXSize
        self.ysize = self.inDataset.RasterYSize
        self.nodata = self.inDataset.GetRasterBand(1).GetNoDataValue()
        self.tileSize = tileSize
        self.batchsize = batch_size
        
    def _load_model(self, weigths):
        _model = MainModel(accelerator=self.accelerator, net_G=   self.unet)
        _model.eval()
        with torch.inference_mode():
            _model.load_state_dict(
                torch.load(weigths, map_location=self.device) )
        return _model

    def _inferTiles(self, tiles:Iterable[GDALtile]):
        n_tiles  = np.array([ np.expand_dims( tile.getArray() /128 -1 , axis=0 ) for tile in tiles ]) 
        T_tiles = torch.tensor(n_tiles, dtype=torch.float16)
        with torch.inference_mode():
            preds = self.model.net_G( T_tiles.to(self.device) )

        C_tiles = lab_to_rgb(T_tiles, preds.cpu())
        for i in range(len(tiles)):
            tiles[i].setArray( (C_tiles[i] * 255).astype(np.uint8) )
        return tiles

    def _process_tiles(self) -> np.ndarray:
        full_img= np.zeros( (3, self.ysize ,self.xsize) , dtype=np.uint8  ) 

        c = 0
        tot = ((self.xsize//self.tileSize)+1)*((self.ysize//self.tileSize)+1)
        for tiles in self._tileGenerator():
            print(f'{c} of {tot} tiles', end='\r')
            newTiles = self._inferTiles(tiles)
            c += len(tiles)
            
            for newTile in newTiles:
                xoff = newTile.posx
                yoff = newTile.posy
                patch = newTile.getArray(self.nodata)
                
                if newTile.pady > 0:
                    patch = patch[0:-1 * newTile.pady, :]
                if newTile.padx > 0:
                    patch = patch[:, 0:-1* newTile.padx]

                bil = np.stack((patch[:,:,0], patch[:,:,1], patch[:,:,2]), axis=0).astype(np.uint8)
                full_img[:, yoff: yoff + bil.shape[1] , xoff: xoff + bil.shape[2] ] = bil

        return full_img

    def _tileGenerator(self) -> Iterable[GDALtile]:
        tiles = []
        c=0
        for xi, xoff in enumerate( range(0, self.xsize , self.tileSize)):
            for yi, yoff in  enumerate( range(0, self.ysize , self.tileSize)):
                c+=1
                # Last row and column are smaller then needed for inference                
                xsize, ysize = (self.tileSize,self.tileSize)
                if (self.xsize - (xi*self.tileSize)) < self.tileSize:
                    xsize =  self.xsize - (xi*self.tileSize)
                if (self.ysize - (yi*self.tileSize)) < self.tileSize:
                    ysize =  self.ysize - (yi*self.tileSize)

                img = self.inDataset.GetRasterBand(1).ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xsize, win_ysize=ysize)
                img = np.pad(img, ((0, self.tileSize- ysize),(0, self.tileSize- xsize)),  mode='median') 

                mask = None
                if self.nodata is not None:  #replace nodata 
                    mask = img == self.nodata
                    img[img == self.nodata] = np.median( img[img != self.nodata] ).astype("uint8")

                tiles.append( GDALtile(img, xoff, yoff, self.tileSize- xsize , self.tileSize- ysize, mask) )
        
                if c % self.batchsize == 0:
                    out_tiles = tiles.copy()
                    tiles = []
                    yield out_tiles

        if len(tiles) > 0: #return remaining tiles
            yield tiles    


    def saveOutDataSet(self, out_GDALfile:os.PathLike, outDriver:str='GTiff', 
                       creation_options:Iterable[str]=None):
        print('Started colorisation')
        img_rgb = self._process_tiles() 
        print("\nColorisation Done, Writing to output: ", out_GDALfile)

        if outDriver in ('PNG', 'JPEG', 'JPEG2000') or outDriver is None:
            fname = Path(out_GDALfile)
            wld_ext = fname.suffix[:-2] + fname.suffix[-1] + 'w'
            io.imsave( fname=fname, arr= img_rgb.transpose((1, 2, 0)) )
            with open( fname.parent / f"{fname.stem}{wld_ext}" ,'w') as wld:
                wld.write("\n".join([str(self.transform[i]) for i in (1,2,4,5,0,3)]))
            return 

        drv = gdal.GetDriverByName(outDriver)
        ## Options for a Geotiff (.tif) 
        if creation_options is None and outDriver == 'GTiff':
            creation_options = [ 'BIGTIFF=YES', 'INTERLEAVE=BAND', 'COMPRESS=DEFLATE',
                    'PREDICTOR=2', 'ZLEVEL=9','Tiled=YES', 
                    f'BLOCKXSIZE={self.tileSize}', f'BLOCKYSIZE={self.tileSize}', 
                    'SPARSE_OK=True', 'NUM_THREADS=ALL_CPUS'  ]
        ## Options for a GeoPackage (.gpkg) 
        elif creation_options is None and outDriver == 'GPKG':
            creation_options = [f'RASTER_TABLE={Path(out_GDALfile).stem}', 
                    'RESAMPLING=LANCZOS', 'TILE_FORMAT=PNG_JPEG', 
                    f'BLOCKSIZE={self.tileSize}', 'TILING_SCHEME=GoogleMapsCompatible' ]
        ## Options for a Cloud Optimized GeoTIFF Generator (.tif)
        elif creation_options is None and outDriver == 'COG':
            creation_options = ['TILING_SCHEME=GoogleMapsCompatible', 'COMPRESS=JPEG',
                    'BIGTIFF=YES', f'BLOCKSIZE={self.tileSize}', 'WARP_RESAMPLING=LANCZOS',
                    'SPARSE_OK=True', 'NUM_THREADS=ALL_CPUS', 'STATISTICS=YES' ]

        self.outDataset = drv.Create(str(out_GDALfile), bands=3, xsize=self.xsize, ysize=self.ysize,
                                                eType=gdal.GDT_Byte, options=creation_options) 
        self.outDataset.WriteArray(img_rgb)
        self.outDataset.SetGeoTransform(self.transform)
        self.outDataset.SetProjection(self.inDataset.GetProjection())
        for b in range(1,4):
            self.outDataset.GetRasterBand(b).SetNoDataValue(self.nodata)
        # free up memory
        self.outDataset.FlushCache()
        self.outDataset = None
        self.inDataset = None