import numpy as np
from pathlib import Path
import xarray as xr
import pandas as pd
from pyproj import Proj, transform
import pickle as pkl
from autocorr_functions import *
import autocorr_cmls as accml
import sys
sys.path.append("/home/adameshel/Documents/code/") 
from helper_functions import *

# Create a dict with aggregation times, time stamps, pars and std errors
# based on the godness of fit to a 2d ACF from a gridded radar.

my_path = Path('/home/adameshel/Documents/code/autocorr/semi_real/main_with_gamma/')

agg_times = ['5T','15T','30T','60T','90T','120T','180T']

raw_path = Path('/home/adameshel/Documents/code/kit_code/\
2d_method_intercomparison/data/raw/')

list_of_GT_datasets = []
ds_radolan = xr.open_mfdataset(
    str(raw_path.joinpath('radklim_yw_for_adam.nc').absolute()),
                              combine='by_coords'
                              )

start_time_idx = 0#15
end_time_idx = -1#70#340#len(ds_radolan_cut.time)

####### CHANGE DOMAIN ######
## Medium cut
min_lat = 47.6890
min_lon = 8.1873
max_lat = 49.1185
max_lon = 10.0978

ds_radolan_cut = ds_radolan.where((ds_radolan['latitudes'] >= min_lat) &
                                 (ds_radolan['latitudes'] <= max_lat) & 
                                 (ds_radolan['longitudes'] >= min_lon) &
                                 (ds_radolan['longitudes'] <= max_lon),
                                 drop=True)

proj_degrees = Proj(init='epsg:4326')
proj_meters = Proj(init='epsg:3043')#3857 FROM MAX  #3395 #3043 UTM

# from pyproj import Transformer
x_grid_utm, y_grid_utm = transform(proj_degrees, 
                         proj_meters, 
                         ds_radolan_cut.longitudes.values, 
                         ds_radolan_cut.latitudes.values)

ds_radolan_cut.coords['x_utm'] = (('y', 'x'), x_grid_utm)
ds_radolan_cut.coords['y_utm'] = (('y', 'x'), y_grid_utm)

time_frame = ds_radolan_cut.time[start_time_idx:end_time_idx]
num_of_ts = len(time_frame)
ds_radolan_GT = ds_radolan_cut.where(ds_radolan_cut.time == \
             ds_radolan_cut.time[start_time_idx:end_time_idx])
ds_radolan_GT = ds_radolan_GT.rename({'rainfall_amount':'raindepth'})

rain_mat = ds_radolan_GT.raindepth.values #12 # to make it mm/h
# rain_mat[rain_mat < 0.1] = 0.0
rain_mat = rain_mat
del ds_radolan_GT
ds_radolan_GT = xr.Dataset(
    data_vars={'raindepth': (('time','y', 'x'), rain_mat)},
    coords={'lon_grid': (('y', 'x'), ds_radolan_cut.longitudes.values),
            'lat_grid': (('y', 'x'), ds_radolan_cut.latitudes.values),
            'x_utm': (('y', 'x'), ds_radolan_cut.x_utm.values),
            'y_utm': (('y', 'x'), ds_radolan_cut.y_utm.values),
            'time': time_frame,
            'x': ds_radolan_cut.x.values,
            'y': ds_radolan_cut.y.values})
d_run = {}

for at, agg in enumerate(agg_times):
    l_pars = []; l_pars_err = []; l_ts = []; l_r2 = []
    print(str("ds_radolan_GT_" + agg))
    globals()["ds_radolan_GT_" + agg] = ds_radolan_GT.resample(
        time=agg, label='right', 
        restore_coord_dims=False).sum(dim='time')
    list_of_GT_datasets.append(str("ds_radolan_GT_" + agg))
    
    for ts in range(len(globals()["ds_radolan_GT_" + agg].time.values)):
        try:
            Z = globals()["ds_radolan_GT_" + agg].raindepth.isel(time=ts).values
            if np.sum(np.isnan(Z)) + np.sum(np.isinf(Z))==0:
                ac = accml.Autocorr(Z)
                ac(optimize=True)
                if round(ac.pars[1],1)!=1.5:
                    r2 = round(np.corrcoef(ac.ac_2d.ravel(),
                                           ac.s.ravel())[1,0],3)**2
                    if r2>=0.6:
                        l_pars_err.append(ac.std_error)
                        l_pars.append(ac.pars) 
                        l_ts.append(ts)
                        l_r2.append(r2) 
                        print(ts,ac.pars,ac.std_error,'r2=%.3f'%r2)
                    d_run[agg] = [l_ts,l_pars,l_pars_err,l_r2]
        except:
            pass
with open(str(my_path.joinpath('d_run.pkl')), 'wb') as f:
    pkl.dump(d_run, f)