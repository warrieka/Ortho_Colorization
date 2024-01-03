from pathlib import Path
from model.GDALfile_colorizer import GDALfile_colorizer

IMAGE_SIZE = 512
MODEL = Path(r".\runs\models\run26\color_run26_512_epoch16.pth").resolve() #Path(r".\runs\models\color_run26_512.pth").resolve()
IN_FILES = r"W:\1968\*.tif"
OUT_DIR =  r"W:\1968_rbg"
BATCH_SIZE = 16
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
    infiles = Path(IN_FILES)
    for gdFile in infiles.parent.glob( infiles.name ):
        outFile = Path(OUT_DIR) / f'{gdFile.stem}{GDAL_FORMATS[GDAL_DRV]}'
        gfc = GDALfile_colorizer(gdFile, MODEL, IMAGE_SIZE, BATCH_SIZE)
        gfc.saveOutDataSet(outFile, GDAL_DRV)
