'''Data Downloader'''
import os
import re
import tqdm
import toml
import argparse
import requests
import datetime
import logging

import openeo
import pyproj
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin, from_bounds

import numpy as np
import pandas as pd
import geopandas as gpd

from utils import *

def linestring_to_raster(
    linestring, 
    width, 
    height, 
    resolution, 
    buffer = 0, 
    x_offset = 0,
    y_offset = 0,
    random_offset = False, 
    random_state = 42
):
    '''
    Rasterizes a linesting to a binary mask.

    Parameters
    ----------
    linestring : shapely.Linestring
        Linestring geometry

    width : int
        Width of the output raster

    height : int
        Height of the output raster

    resolution : int
        Resolution of the output raster

    buffer : int
        Half-width of runway to detect. This is relative to the CRS of the linestring.

    x_offset : int
        Offset distance in x direction; relative to image size

    y_offset : int
        Offset distance in y direction; relative to image size

    random_offset : bool
        If `True`, applies a random translation
        
    random_state : int
        Seed for random_offset

    Returns
    -------
    nd.array of size (height, width)
    '''
    # set bounds
    bounds = linestring.bounds
    bounds = [floor10(bounds[0]), floor10(bounds[1]), ceil10(bounds[2]), ceil10(bounds[3])]

    # add buffer for thicker mask
    if buffer:
        linestring = linestring.buffer(buffer)

    # get image original size
    y_size = bounds[3] - bounds[1]
    x_size = bounds[2] - bounds[0]

    # estimate difference between linestring bounds and desired output size
    y_bounds_diff = height * resolution - y_size
    x_bounds_diff = width * resolution - x_size

    # select random offsets
    if random_offset:
        np.random.seed(random_state)
        y_offset = np.random.randint(-0.75 * height * resolution, 0.75 * height * resolution, 1)
        x_offset = np.random.randint(-0.75 * width * resolution, 0.75 * width * resolution, 1)
    else:
        y_offset = y_offset * resolution
        x_offset = x_offset * resolution

    # calculate shifts
    y_1 = floor10(int(0.5 * y_bounds_diff) + y_offset)
    y_2 = y_bounds_diff - y_1 
    x_1 = floor10(int(0.5 * x_bounds_diff) + x_offset)
    x_2 = x_bounds_diff - x_1

    # define new bounds
    # create base raster
    # compute new transforms
    new_bounds = [bounds[0] - x_1, bounds[1] - y_1, bounds[2] + x_2, bounds[3] + y_2]
    raster_array = np.zeros((height, width), dtype = 'uint8')
    transform = from_bounds(*new_bounds, *raster_array.T.shape)

    # rasterize layer
    rasterized = rasterize(
        [linestring],
        out_shape = raster_array.shape,
        transform = transform,
        fill = 0,
        all_touched = True,
        dtype='uint8'
    )
    return rasterized, new_bounds, transform

def download_sdata(
    connection, 
    collection, 
    bands, 
    bbox, 
    temporal_extent, 
    crs, 
    resolution, 
    max_cloud_cover, 
    basename,
):
    ''' Downloads satellite data from coperniucs using openeo
    Parameters
    ----------
    connection : openeo.Connection
        An openeo connection
        
    collection : str
        Name of copernicus collection to download

    bands : list
        List of bands to download

    bbox : list
        List containing extents in xmin, ymin, xmax, ymax format

    temporal_extent : list
        List of date strings specifying start and end date.

    resolution : int
        Spatial resolution of data download
        
    max_cloud cover: int
        Integer between 0 and 100 to filter data based on cloud cover

    basename : str
        Basename of the file. _sdata.tif will be appended to this to create output name

    Returns
    -------
    None
    '''
    spatial_extent = {
        "west": bbox[0],
        "south": bbox[1],
        "east": bbox[2],
        "north": bbox[3],
        'crs' : crs
    }
    
    datacube = connection.load_collection(
        collection_id = collection,
        spatial_extent = spatial_extent,
        temporal_extent = temporal_extent,
        bands = bands,
        max_cloud_cover = max_cloud_cover
    )

    datacube = datacube.resample_spatial(
        resolution = resolution,  # Resolution in meters for UTM
        projection = crs  # UTM Zone 33N
    )
    scl_band = datacube.band("SCL")
    mask = (scl_band == 1) & (scl_band == 2) & (scl_band == 3) & (scl_band == 7) & (scl_band == 8) & (scl_band == 9)
    mask_resampled = mask.resample_cube_spatial(datacube.band('B02'))
    datacube_masked = datacube.mask(mask_resampled)
    datacube_composite = datacube_masked.min_time()
    sdata_name = f'{basename}_sdata.tiff'

    # save download
    datacube_composite.download(sdata_name, format = 'GTiff')

def create_data_sample(df, connection, collection, bands, temporal_extent, width, height, resolution, crs, max_cloud_cover, buffer, **kwargs):
    '''Creates an image mask data sample in a folder. 
    Parameters
    ----------
    df : pd.series
        Row data from a pandas dataframe
    connection : oidc connection
        An OIDC connection instance
    collection : str
        Name of the collection to use.
    bands : list
        List containing bands to extract.
    temporal_extent : list or tuple
        Time range to download from; format = (start, end).
    width : int
        Output width
    height : int
        Output height
    resolution : int
        Scale to download at
    crs : str
        EPSG projection
    max_cloud_cover : int
        Maximum cloud cover allowed per image

    Returns
    -------
    None
    '''
    
    os.makedirs(df.output_dir, exist_ok = True)
    name = os.path.join(df.output_dir, df.filename)
    geom = df.geometry

    # compute binary_mask
    binary_mask, bounds, transform = linestring_to_raster(
        geom, 
        buffer = buffer,
        width = width, 
        height = height, 
        resolution = resolution,
        x_offset = df.x_offset, 
        y_offset = df.y_offset, 
        random_offset = False, 
        **kwargs
    )

    # save binary mask and sdata
    mask_name = f'{name}_target.tiff'

    with rasterio.open(
        mask_name,
        'w',
        driver = 'GTiff',
        height = binary_mask.shape[0],
        width = binary_mask.shape[1],
        count = 1,
        dtype = rasterio.uint8,
        crs = crs ,
        transform = transform,
    ) as dst:
        dst.write(binary_mask, int(1))
    print('data downloaded')
    
    download_sdata(
        connection = connection, 
        collection = collection, 
        bbox = bounds,
        bands = bands, 
        temporal_extent = temporal_extent,
        crs = crs,
        resolution = resolution, 
        max_cloud_cover = max_cloud_cover,
        basename = name
    )

def expand_df(df, offset, data_dir = None, random_state = 0):
    '''Add additional information to the train dataframe
    Parameters
    ---------
    df : pd.DataFrame
        Dataframe to expand.
    offset : int
        Offset distance for adjacent non-runway images
    data_dir : str or path
        Root data folder
    random_state : int
        Random seed for shuffling the dataset
    
    Returns
    -------
    pd.DataFrame : A new dataframe with additional columns
    '''
    df1 = df.copy()
    df2 = df.copy()
    df1['runway'] = 1
    df1['x_offset'] = 0
    df1['y_offset'] = 0

    offsets = [-offset, -offset // 2, offset // 2, offset]
    np.random.seed(random_state)
    df2['runway'] = 0
    df2['Activo'] = 2
    df2['x_offset'] = np.random.choice(offsets, size = len(df2))
    df2['y_offset'] = np.random.choice(offsets, size = len(df2))

    new_df = pd.concat([df1, df2]).sample(frac = 1, replace = False, random_state = random_state).sort_values('id').reset_index(drop = True)
    if data_dir is not None:
        new_df['output_dir'] = new_df.apply(lambda x : os.path.join(data_dir, f'{x.id}_{x.yr}_{x.runway}'), axis = 1)
    else:
        new_df['output_dir'] = new_df.apply(lambda x : f'{x.id}_{x.yr}_{x.runway}', axis = 1)
    new_df['filename'] = new_df.apply(lambda x: f'{x.id}_{x.yr}_active{x.Activo}_runway{x.runway}', axis = 1)

    return new_df

def get_start_date(lon, lat, start_date, end_date, box, cloud_cover):
    '''
    Extracts available start date with the maximum cloud cover specified.
    lon : float
        Longitude
    lat : float
        Latitude
    start_date : str
        Start date
    end_date : str
        End date
    box : list
        Bounds of region to query
    cloud_cover : int
        Maximum allowed cloud cover
    Returns
    -------
    None
    '''
    json = requests.get(f'https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel2/search.json?productType=S2MSI2A&cloudCover=[0,{cloud_cover}]&startDate={start_date}&completionDate={end_date}&lon={lon}&lat={lat}&box={",".join([str(i) for i in box])}').json()
    output = pd.DataFrame.from_dict(json['features'])
    if len(output) > 1:
        dates = [i['startDate'] for i in output.properties]
        return min(dates)
    return None

def create_date(date_str):
    '''Constructs datetime object from date str'''
    Y, m, d = [int(i) for i in date_str.split('-')]
    date = datetime.date(Y, m, d)
    return date

def get_available_dates(df, date, timedelta, crs, max_cloud_cover):
    '''Incrementally searches for availabe data from specified date range.
    Parameters
    ----------
    df : pd.series
        A dataframe row
    date : str
        Date to query from
    timedelta : int
        Step size (in days) for searching
    crs : int
        EPSG format crs 
    max_cloud_cover : int
        Maximum cloud cover 
    Returns
    -------
    tuple : (start, end)
    '''
    # format date
    year = int(date.split('-')[0])
    date = create_date(date)
    start_date = date - datetime.timedelta(timedelta) # get start date
    end_date = date + datetime.timedelta(timedelta) # get start date

    # format bounds
    lon, lat = df.geometry.bounds[:2] # get coordinates
    lon2, lat2 = df.geometry.bounds[2:]
    projector = pyproj.Transformer.from_crs(crs, 'epsg:4326')
    lat, lon = projector.transform(lon, lat)
    lat2, lon2 = projector.transform(lon2, lat2)
    box = [lon, lat, lon2, lat2]
    
    status = None # initialize available_start_date
    
    # while loop till it gets the data, check as far back as 2015
    while status is None:
        end = end_date.strftime('%Y-%m-%d')#T%H:%M:%SZ')
        start = start_date.strftime('%Y-%m-%d')#T%H:%M:%SZ')
        available_start_date = get_start_date(lon, lat, start, end, box, max_cloud_cover)
        if available_start_date is not None:
            status = 'Found'
            break
        start_date = start_date - datetime.timedelta(timedelta)
        end_date = end_date + datetime.timedelta(timedelta)
        if start_date.year == 2014 and start_date.month == 12:
            break
        if end_date > datetime.date.today():
            end_date = datetime.date.today()
        if start_date < datetime.date(year, 1, 1) and end_date == datetime.date.today():
            break
    if available_start_date is None:
        start_date = None
        end_date = None
    else:
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    
    return start_date, end_date
    
def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help = 'path to config file', required = True)
    parser.add_argument('-t', '--type', help = 'type of data, train or test', required = False, default = 'train')
    parser.add_argument('-dt', '--timedelta', help = 'Days to look behind to find data', required = False, type = int, default = 30)
    args = parser.parse_args()
    
    # get config and load files
    config = toml.load(args.config)

    # defaults
    LOGDIR = config['output']['logs_dir']
    URL = config['params']['data']['sentinel_url']
    COLLECTION = config['params']['data']['sentinel_collection']
    BANDS = config['params']['data']['sentinel_bands']
    MAX_CLOUD_COVER = config['params']['data']['sentinel_max_cloud_cover']
    RESOLUTION = config['params']['data']['resolution']
    CRS = config['params']['spatial']['output_crs']
    GEOG_CRS = config['params']['spatial']['geographic_crs']

    # Initialize logger
    os.makedirs(LOGDIR, exist_ok = True)
    global logger
    logging.basicConfig(
        filename = os.path.join(LOGDIR, f'{args.type}_download.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'  # Append to the log file
    )
    logger = logging.getLogger(__name__)
    logger.info(f'Initializing download with {args.config}')
    logger.info(f'Params: timedelta: {args.timedelta}, crs : {CRS}, resolution: {RESOLUTION}, url: {URL}, collection: {COLLECTION}, bands: {BANDS}, max_cloud_cover: {MAX_CLOUD_COVER}')

    csv_file = os.path.join(LOGDIR, f'{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}_{args.type}_params.csv')
    try:
        with open(csv_file, 'w') as file:
            if file.tell() == 0:
                file.write('name,start_date,end_date,status\n')
                file.close()
        logger.info(f'Initializing {csv_file} succeeded!') 
    except Exception as e:
        logger.error(f'failed to initializing {csv_file}') 

    connection = openeo.connect(url = URL)
    connection.authenticate_oidc()

    if args.type == 'train':
        source_dir = config['data']['raw_dataset']['train_data_dir']       
        output_dir = config['data']['image_data']['train_image_dir']
        WIDTH = config['params']['data']['width']
        HEIGHT = config['params']['data']['height']
        BUFFER = config['params']['data']['linestring_buffer']
        logger.info(f'Image params: width: {WIDTH}, height: {HEIGHT}, buffer: {BUFFER}')
        
        gdf = gpd.read_file(source_dir)
        gdf = expand_df(gdf, WIDTH, data_dir = output_dir)
        gdf = gdf.to_crs(CRS) 
        count = 0
        for _, data in tqdm.tqdm(gdf.iterrows(), total = len(gdf)):
            # get start and end dates to download from
            logger.info(f'Starting download, file: {data.filename}')
            try:
                date = f'{data.yr}-12-31'
                start_date, end_date = get_available_dates(
                    df = data, 
                    date = date, 
                    timedelta = args.timedelta, 
                    crs = CRS,
                    max_cloud_cover = MAX_CLOUD_COVER
                )
                
                # condition to stop if no data available
                if  start_date is None:
                    print(f'No data found for {data.id}, Skipping!')
                    pass
    
                # construct time range
                start_date, end_date = None, None
                temporal_extent = [start_date, end_date]
                logger.info(f'temporal_extent : {temporal_extent[0]}, {temporal_extent[1]}')
    
                # download sample
                create_data_sample(
                    df = data, 
                    connection = connection, 
                    collection = COLLECTION, 
                    bands = BANDS,
                    temporal_extent = temporal_extent, 
                    width = WIDTH,
                    height = HEIGHT,
                    resolution = RESOLUTION,
                    crs = CRS,
                    max_cloud_cover = MAX_CLOUD_COVER,
                    buffer = BUFFER
                )
                logger.info(f'{data.filename}_sdata.tiff and {data.filename}_target.tiff successfully saved to {data.output_dir}')

                # update csv file
                with open(csv_file, 'a') as file:
                    file.write(f'{data.filename},{temporal_extent[0]},{temporal_extent[1]},1\n')
                logger.info(f'temporal extent saved to {csv_file}')
                count += 1

            except Exception as e:
                print(e)
                logger.error(f'Error occured while downloading {data.filename}:{e}')
                with open(csv_file, 'a') as file:
                    file.write(f'{data.filename},{temporal_extent[0]},{temporal_extent[1]},0\n')
                logger.info(f'temporal extent saved to {csv_file}')
            
    elif args.type == 'test':
        source_dir = config['data']['raw_dataset']['test_data_dir']
        output_dir = config['data']['image_data']['test_image_dir']
        parse_test_files(source_dir) #breaks all the shape files into different folders
        filenames = os.listdir(source_dir)
        years = [i.split('_')[1] for i in filenames]
        test_shp_files = [os.path.join(source_dir, i) for i in os.listdir(source_dir)]
        gdf = pd.concat([ gpd.read_file(i) for i in test_shp_files])
        gdf['filename'] = filenames
        gdf['year'] = years
        
        # download the sdata
        os.makedirs(output_dir, exist_ok = True)
        for _, data in tqdm.tqdm(gdf.iterrows(), total = len(gdf)):
            # get start and end dates to download from
            logger.info(f'Starting download, file: {data.filename}')
            try:
                date = f'{data.year}-12-31'
                start_date, end_date = get_available_dates(
                    df = data, 
                    date = date, 
                    timedelta = args.timedelta, 
                    crs = CRS,
                    max_cloud_cover = MAX_CLOUD_COVER
                )
    
                # condition to stop if no data available
                if  start_date is None:
                    print(f'No data found for {data.id}, Skipping!')
                    pass
    
                # construct time range
                temporal_extent = [start_date, end_date]
                logger.info(f'temporal_extent : {temporal_extent[0]}, {temporal_extent[1]}')
    
                # download data
                name = os.path.join(output_dir, data.filename)
                download_sdata(
                    connection = connection, 
                    collection = COLLECTION, 
                    bbox = data.geometry.bounds,
                    bands = BANDS, 
                    temporal_extent = temporal_extent,
                    crs = CRS,
                    resolution = RESOLUTION, 
                    max_cloud_cover = MAX_CLOUD_COVER,
                    basename = name
                )
                logger.info(f'{data.filename}_sdata.tiff and {data.filename}_target.tiff successfully saved to {output_dir}')
                
                # update csv file
                with open(csv_file, 'a') as file:
                    file.write(f'{data.filename},{temporal_extent[0]},{temporal_extent[1]},1\n')
                logger.info(f'temporal extent saved to {csv_file}')
            except Exception as e:
                print(e)
                logger.error(f'Error occured while downloading {data.filename}:{e}')
                with open(csv_file, 'a') as file:
                    file.write(f'{data.filename},{temporal_extent[0]},{temporal_extent[1]},0\n')
                logger.info(f'temporal extent saved to {csv_file}')
    else:
        raise ValueError(f'Unrecognized type - `{args.type}`. Please specify one of `train` or `test`')

    gdf.to_file(os.path.join(output_dir, args.type + '.geojson'), driver='GeoJSON')
    logger.info(f'{args.type}.geojson saved!')

if __name__ == '__main__':
    main()