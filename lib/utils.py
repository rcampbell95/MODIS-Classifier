def load_data(bands, path):
    import gdal
    import numpy as np
    import os

    dirpath, dirname, filenames = next(os.walk(path))
    
    num_bands = 10

    data = []
    for file in filenames:
        bands = []
        image = gdal.Open(dirpath + "/" + file)
        
        for band_i in range(num_bands):
            rband = image.GetRasterBand(band_i + 1)
            # Fill nans in data
            bands.append(np.nan_to_num(rband.ReadAsArray()))
            
        data.append(bands)

    return np.array(data), filenames