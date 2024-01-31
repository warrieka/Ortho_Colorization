from pathlib import Path
import argparse
from model.GDALfile_colorizer import GDALfile_colorizer

IMAGE_SIZE = 512
ARCHITECTURE = "resnet34"
MODEL = Path(r"runs\models\run29\color_run29_resnet34_512_epoch52.pth").resolve() #Path(r".\runs\models\color_run26_512.pth").resolve()
IN_FILES = r"V:\project\histo\moz\antwerp_luchtfoto1940_1944_4.tif"
OUT_DIR =  r"V:\project\histo\rgb"
#Optional, defaults to geotiff: 
GDAL_DRV = 'GTiff'
#supported output formats  by GDAL driver
GDAL_FORMATS = {
    "GTiff" :".tif", #Regular Geotiff
    "PNG":".png",   #Portable Network Graphics, georefencing stored in worldfile
    "JPEG":".jpg",  #Joint Photography Experts Group JFIF File Format, georefencing stored in worldfile
    "JPEG2000":".jp2", #Joint Photography Experts, JPEG2000 based on OpenJPEG library v2
    "COG": ".tif",  #Cloud Optimized GeoTIFF generator: https://www.cogeo.org/
    "GRIB2" :".grb", #GRIB is commonly used for distribution by the World Meteorological Organization
    "HDF4Image":".hdf", #Hierarchical Data Format, like NASA's Earth Observing System (EOS) HDF4-EOS
    "SAGA" :".sdat" , #System for Automated Geoscientific Analyses, Binary Grid File Format
    "MBTiles" :".sqlite", #MapBox Tile format, https://docs.mapbox.com/help/glossary/mbtiles/
    "TileDB": '', #TileDB raster, see https://tiledb.com/
    "GPKG" :".gpkg", #Rastertiles stored in OGC Geopackage (based on sqlite), see https://www.geopackage.org/
    "R" :".rda" ,  #R Object Data Store, see https://www.r-project.org/
    "NUMPY": '.npy' } #numpy's binary matrix format https://numpy.org/doc/stable/reference/routines.io.html


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description=
      'Deeplearing model to colorize black&white orthophoto\'s.\n'+
      'This tool let you use the model to colorize a gdal readable files.')
    
    parser.add_argument('--input', default=IN_FILES, type=Path, 
        help='The input file(s), you can use a glob expression like "*.tif" to specify multiple files')
    
    parser.add_argument('--out_dir', default=OUT_DIR, type=Path, 
        help='The output location of the resulting colorisation, don\'t use the input folder!.')
        
    parser.add_argument('--out_driver', default=GDAL_DRV,
        help='The output gdal driver to use to write output,\n'+
        'only drivers that support "create" can be used (see https://gdal.org/drivers/raster/)')
    
    parser.add_argument('--batch_size', default=16, type=int, 
        help='the size of batch the algoritem sends to the GPU in 1 batch, \n'+
        'If you get CUDA of of memory issues, try to decreaser the batch')

    opts = parser.parse_args()

    infiles = opts.input.resolve()
    out_dir = opts.out_dir.resolve()
    drv = opts.out_driver

    for gdFile in infiles.parent.glob( infiles.name ):
        outFile = out_dir / f'{gdFile.stem}{GDAL_FORMATS[drv]}'
        gfc = GDALfile_colorizer(gdFile, MODEL, IMAGE_SIZE, opts.batch_size, ARCHITECTURE)
        gfc.saveOutDataSet(outFile, drv)
