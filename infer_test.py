from pathlib import Path
from model.GDALfile_colorizer import GDALfile_colorizer

IMAGE_SIZE = 224
MODEL =  Path(f'.\\runs\\models\\color_run18_0.60m_{IMAGE_SIZE}.pth').resolve()
IN_FILES = r"V:\project\histo\tests\in\*.png"
OUT_DIR = r"V:\project\histo\tests\out"
MAX_CONTRAST = False
#Optional, defaults to geotiff: 
GDAL_DRV = 'PNG'
#supported output formats  by GDAL driver
GDAL_FORMATS = {"GTiff" :".tif", #Regular Geotiff
    "PNG":".png",   #Portable Network Graphics
    "JPEG":".jpg",  #Joint Photography Experts Group JFIF File Format
    "JPEG2000":".jp2", #Joint Photography Experts, JPEG2000 based on OpenJPEG library v2
    "COG": ".tif",  #Cloud Optimized GeoTIFF generator: https://www.cogeo.org/
    "GRIB2" :".grb", #GRIB  is commonly used for distribution by the World Meteorological Organization
    "HDF4Image":".hdf", #Hierarchical Data Format, like NASA's Earth Observing System (EOS) HDF4-EOS
    "SAGA" :".sdat" , #System for Automated Geoscientific Analyses, Binary Grid File Format
    "MBTiles" :".sqlite", #MapBox Tile format, https://docs.mapbox.com/help/glossary/mbtiles/
    "TileDB": '', #TileDB raster, see https://tiledb.com/
    "GPKG" :".gpkg", #Rastertiles stored in OGC Geopackage, see https://www.geopackage.org/
    "R" :".rda" ,  #R Object Data Store, see https://www.r-project.org/
    "NUMPY": '.npy' } #numpy's raster format https://numpy.org/doc/stable/reference/routines.io.html


if __name__ == '__main__':
    infiles = Path(IN_FILES)
    for gdFile in infiles.parent.glob( infiles.name ):
        outFile = Path(OUT_DIR) / f'{gdFile.stem}{GDAL_FORMATS[GDAL_DRV]}'
        gfc = GDALfile_colorizer(gdFile, MODEL, IMAGE_SIZE)
        gfc.saveOutDataSet(outFile, GDAL_DRV)
