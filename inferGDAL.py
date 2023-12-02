from pathlib import Path
from model.GDALfile_colorizer import GDALfile_colorizer

IMSIZE = 512
MODEL =  r".\runs\model_pil_run10_1m_512.pth"
IN_FILES = r"W:\1970\OKZPAN71VL_K15.jp2"
OUT_DIR = r"W:\testdata"

#Optional: 
GDAL_DRV = 'GTiff'
REF_PIC = None   # path to an image who's histgram will be matched.
IMPROVE_CONTRAST = False
# Best options for Geotiff
GDAL_CREATION_OPTS = ['BIGTIFF=YES', 'INTERLEAVE=BAND', 'COMPRESS=JPEG',
                      'Tiled=YES', f'BLOCKSIZE={IMSIZE}', 
                      'SPARSE_OK=True', 'NUM_THREADS=ALL_CPUS' ]

# ## Options for a Cloud Optimized Generator (COG)
# GDAL_CREATION_OPTS = ['TILING_SCHEME=GoogleMapsCompatible', 'COMPRESS=JPEG',
#                       'BIGTIFF=YES', f'BLOCKSIZE={IMSIZE}',
#                       'SPARSE_OK=True', 'NUM_THREADS=ALL_CPUS' ]
# ## ---------------------------------------------------- ## 
# ## Options for a GeoPackage (.gpkg) 
# ## Change line 40 to: `outFile = OUT_DIR + ".gpkg"`
# GDAL_CREATION_OPTS = [f'RASTER_TABLE={Path(gdFile).stem}', 'RESAMPLING=CUBIC',
#                       'TILE_FORMAT=PNG_JPEG', f'BLOCKSIZE={IMSIZE}', 
#                       'TILING_SCHEME=GoogleMapsCompatible' ]
# ## ---------------------------------------------------- ## 
GDAL_FORMATS = {"GTiff" :".tif", "PNG":".png", "JPEG":".jpg", 
                "HDF4Image":".hdf", "OpenFileGDB": ".gdb", "SAGA" :".sdat" ,
                "JPEG2000":".jp2", "GRIB" :".grb", "COG": ".tif",
                "MBTiles" :".sqlite", "TileDB": '', "GPKG" :".gpkg",
                "R" :".rda" ,  "NUMPY": '.npy' }


if __name__ == '__main__':
    infiles = Path(IN_FILES)
    for gdFile in infiles.parent.glob( infiles.name ):
        outFile = Path(OUT_DIR) / f'{gdFile.stem}{GDAL_FORMATS[GDAL_DRV]}'
        gfc = GDALfile_colorizer(gdFile, MODEL, IMSIZE, REF_PIC)
        gfc.saveOutDataSet(outFile, GDAL_DRV, GDAL_CREATION_OPTS)
