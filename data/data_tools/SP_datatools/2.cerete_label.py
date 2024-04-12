
import os
import numpy as np
from scipy.ndimage.morphology import *
from tqdm import trange
from osgeo import gdal, osr, ogr

def ConvertToRoadSegmentation(tif_file, geojson_file, out_file, isInstance=False):
    # Read Dataset from geo json file
    dataset = ogr.Open(geojson_file)
    if not dataset:
        print('No Dataset')
        return -1
    layer = dataset.GetLayerByIndex(0)
    if not layer:
        print('No Layer')
        return -1

    # First we will open our raster image, to understand how we will want to rasterize our vector
    raster_ds = gdal.Open(tif_file, gdal.GA_ReadOnly)

    # Fetch number of rows and columns
    ncol = raster_ds.RasterXSize
    nrow = raster_ds.RasterYSize

    # Fetch projection and extent
    proj = raster_ds.GetProjectionRef()
    ext = raster_ds.GetGeoTransform()

    raster_ds = None

    # Create the raster dataset
    memory_driver = gdal.GetDriverByName('GTiff')
    out_raster_ds = memory_driver.Create(out_file, ncol, nrow, 1, gdal.GDT_Byte)

    # Set the ROI image's projection and extent to our input raster's projection and extent
    out_raster_ds.SetProjection(proj)
    out_raster_ds.SetGeoTransform(ext)

    # Fill our output band with the 0 blank, no class label, value
    b = out_raster_ds.GetRasterBand(1)

    if isInstance:
        b.Fill(0)
        # Rasterize the shapefile layer to our new dataset
        status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                     [1],  # output to our new dataset's first band
                                     layer,  # rasterize this layer
                                     None, None,  # don't worry about transformations since we're in same projection
                                     [0],  # burn value 0
                                     ['ALL_TOUCHED=TRUE',  # rasterize all pixels touched by polygons
                                      'ATTRIBUTE=road_type']  # put raster values according to the 'id' field values
                                     )
    else:
        b.Fill(0)
        # Rasterize the shapefile layer to our new dataset
        status = gdal.RasterizeLayer(out_raster_ds,  # output to our new dataset
                                     [1],  # output to our new dataset's first band
                                     layer,  # rasterize this layer
                                     None, None,  # don't worry about transformations since we're in same projection
                                     [255]  # burn value 0
                                     )

    # Close dataset
    out_raster_ds = None

    return status

def CreateRoadLabel(tif_file,geojson_file,out_tif_file):

    ## The default image size of Spacenet Dataset is 1300x1300.
    status = ConvertToRoadSegmentation(tif_file, geojson_file, out_tif_file)

    if status != 0:
        print("|xxx-> Not able to convert the file {}. <-xxx".format(geojson_file))
        distance_array = None
    else:
        gt_dataset = gdal.Open(out_tif_file, gdal.GA_ReadOnly)
        if not gt_dataset:
            print('error')
            exit(0)
        gt_array = gt_dataset.GetRasterBand(1).ReadAsArray()

        distance_array = distance_transform_edt(1 - (gt_array / 255))
        std = 15
        distance_array = np.exp(-0.5 * (distance_array * distance_array) / (std * std))
        distance_array *= 255

    return distance_array



def main(root_dir = r"../train/SpaceNet"):

    for _file in os.listdir(root_dir):
        city_file = root_dir.replace("\\",'/') + '/' + _file
        geojson_roads_file = city_file + '/' + 'geojson_roads'
        RGB_roads_file = city_file + '/' + 'PS-RGB'
        out_roads_file = city_file + '/' + 'label_center_line'
        os.makedirs(out_roads_file,exist_ok=True)
        with trange(len(os.listdir(RGB_roads_file))) as t:
            for __file,index in zip(os.listdir(RGB_roads_file),t):
                geojson_file = geojson_roads_file + '/' + __file.replace('_PS-RGB_','_geojson_roads_')\
                                .replace('.tif','.geojson')
                tif_file = RGB_roads_file + '/' + __file
                out_tif_file = out_roads_file + '/' + __file.replace('_PS-RGB_','_roads_label_')
                CreateRoadLabel(tif_file, geojson_file, out_tif_file)

if __name__ == "__main__":
    print('cwd',os.getcwd())
    root_dir = os.getcwd()+"/data/train/SpaceNet"
    main(root_dir)