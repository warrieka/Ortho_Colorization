from pathlib import Path
from model.GDALfile_colorizer import GDALfile_colorizer

IMSIZE = 512
MODEL =  r".\runs\model_pil_run10_1m_512.pth"
IN_FILES = r"W:\1970\OKZPAN71VL_K07.jp2"
OUT_DIR = r"W:\1970_rbg"
GDAL_DRV = 'GTiff'

GDAL_CREATION_OPTS = ['BIGTIFF=YES', 'INTERLEAVE=BAND', 'COMPRESS=JPEG',
                      'Tiled=YES', f'BLOCKXSIZE={IMSIZE}', f'BLOCKYSIZE={IMSIZE}', 
                      'SPARSE_OK=True', 'NUM_THREADS=ALL_CPUS' ]
SUPPORTED_FORMATS = {"GTiff" :".tif" , "AIG" :".None" , "AAIGrid" :".asc" ,
                     "PNG":".png", "JPEG":".jpg", "GIF":".gif", "HDF4":".hdf", 
                     "JP2OpenJPEG":".jp2" ,"GRIB" :".grb" , "R" :".rda" , 
                     "KMLSUPEROVERLAY" :".kml" , "WEBP" :".webp" , "PDF" :".pdf" , 
                     "MBTiles" :".mbtiles" , "HDF5" :".hdf5" , "SAGA" :".sdat" , 
                     "NUMPY": '.npy' }

if __name__ == '__main__':
    infiles = Path(IN_FILES)
    for gdFile in infiles.parent.glob( infiles.name ):
        outFile = Path(OUT_DIR) / f'{gdFile.stem}{SUPPORTED_FORMATS[GDAL_DRV]}'
        gfc = GDALfile_colorizer(gdFile, MODEL, IMSIZE)
        print("Writing to",outFile)
        gfc.saveOutDataSet(outFile, GDAL_DRV, GDAL_CREATION_OPTS)
