import numpy as np
from scipy.linalg.special_matrices import leslie
from pathlib import Path
##
agg_times = ['180T']
identical_l = False
shortest = 1.0; longest = 30.0
num_of_ls = 20
cml_cent_sim = range(50)
mult = 1 # simply for making the rain stronger
ts = 9#2,8,9,11 14 26 49#120#22#3*17 #timestamp
cod = 120 #cutoff distance (km)
aggregation_mean = False
opt = True
bandwidth = 1.0 # km
links_density = 0.03 # km**-2 # original=0.012   0.05
save_cml = True
discard_zeros = False ## Discard zeros from cmls not yet working
l_dist = 'U' # E- exponent, U- uniform, N- none
if identical_l == True: 
    l_dist = 'N'
dir_path = Path('/home/adameshel/Documents/code/autocorr/semi_real/main_with_gamma/')

if identical_l is False:
    diff = longest - shortest
    lag = diff / 5
    shortest = shortest + lag
    longest = longest - lag
cml_lengths = np.linspace(shortest,longest,num_of_ls)

import xarray as xr
import pandas as pd
import random
import string
import glob
import os
from autocorr_functions import *
import autocorr_cmls as accml

raw_path = Path('/home/adameshel/Documents/code/kit_code/\
2d_method_intercomparison/data/raw/')

list_of_datasets = []
ds_radolan = xr.open_mfdataset(
    str(raw_path.joinpath('radklim_yw_for_adam.nc').absolute()),
                              combine='by_coords'
                              )

from pyproj import Proj, transform
import scipy.stats as stats
import sys
from pathlib import Path
# sys.path.append("../Iterative/")
# sys.path.append("../Kriging/")
sys.path.append("/home/adameshel/Documents/code/my_functions/")
from geoFunc import *
import iterative_IDW_V1 as gmz
import kriging as krg
sys.path.append("/home/adameshel/Documents/code/") 
from helper_functions import split_at
import shutil

current = str(str(agg_times[0]) + '_ts' +\
                str(ts) + '_cod' + str(int(cod)) +\
                'opt' + str(opt) + '_identical' +\
                str(identical_l) + '_mult' +\
                str(mult) + '_Dzeros' +\
                str(discard_zeros) +\
                '_'+ str(l_dist))
dir_path_current = dir_path.joinpath(current)
if save_cml==True:
    if os.path.exists(dir_path_current):
        print('Replacing exsisting directory')
        shutil.rmtree(dir_path_current)
    os.makedirs(dir_path_current)

rad_current = str(str(agg_times[0]) + '_ts' +\
                str(ts) + '_cod' + str(int(cod)) +\
                '_mult' +\
                str(mult) + '_Dzeros' +\
                str(discard_zeros))
rad_path_parent = Path(
    '/home/adameshel/Documents/code/autocorr/' +\
        'radar_autocorr_snaps/with_gamma/'
    )
rad_paths = glob.glob(str(rad_path_parent.absolute()) + '/*/')
rad_path_current =  rad_path_parent.joinpath(rad_current)
if str(rad_path_current) + '/' not in rad_paths:
    analyze_radar = True
    os.mkdir(rad_path_current)
else:
    analyze_radar = False

start_time_idx = 0#15
end_time_idx = -1#70#340#len(ds_radolan_cut.time)
############################
####### CHANGE DOMAIN ######
## Medium cut
min_lat = 47.6890
min_lon = 8.1873
max_lat = 49.1185
max_lon = 10.0978

# min_lat = 47.8890
# min_lon = 8.9873
# max_lat = 49.2185
# max_lon = 10.6978

## Interesting cut big rectangle south
# min_lat = 48.000
# min_lon = 8.2000
# max_lat = 50.00
# max_lon = 11.5000

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


# transformer = Transformer.from_proj(proj_degrees, proj_meters)
# x_grid_utm, y_grid_utm = transformer.transform(
#     ds_radolan_cut.longitudes.values, 
#     ds_radolan_cut.latitudes.values
# )

ds_radolan_cut.coords['x_utm'] = (('y', 'x'), x_grid_utm)
ds_radolan_cut.coords['y_utm'] = (('y', 'x'), y_grid_utm)

time_frame = ds_radolan_cut.time[start_time_idx:end_time_idx]
num_of_ts = len(time_frame)
ds_radolan_GT = ds_radolan_cut.where(ds_radolan_cut.time == \
             ds_radolan_cut.time[start_time_idx:end_time_idx])
ds_radolan_GT = ds_radolan_GT.rename({'rainfall_amount':'raindepth'})

import pycomlink as pycml
import pickle as pkl

rain_mat = ds_radolan_GT.raindepth.values #12 # to make it mm/h
# rain_mat[rain_mat < 0.1] = 0.0
rain_mat = rain_mat * mult
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

def cml_lat_lon_to_UTM(df):
    df['xa'], df['ya'] = transform(proj_degrees, 
                                   proj_meters, 
                                   df['site_a_longitude'].values, 
                                   df['site_a_latitude'].values)
    df['xb'], df['yb'] = transform(proj_degrees, 
                                   proj_meters, 
                                   df['site_b_longitude'].values, 
                                   df['site_b_latitude'].values)
    return df

def link_id_gen(
    num_of_ids=1, size=8, chars=string.ascii_uppercase + string.digits
    ):
    my_id_list = list()
    for i in range(num_of_links):
        my_id = ''.join(random.choice(chars) for _ in range(size))
        my_id_list.append(my_id[:4] + '-' + my_id[4:])
    return my_id_list
 
km_in_start = 11
km_in_end = -12
x1 = ds_radolan_GT.x_utm.values[km_in_start:km_in_end]
y1 = ds_radolan_GT.y_utm.values[km_in_start:km_in_end]

for il, l in enumerate(cml_lengths):
    length_name = round(l,1)
    length_name = split_at(str(format(length_name/100, '.3f')),'.',1)[-1]
    globals()['ac_par_il_' + length_name] = np.array([999,999,999])
    if save_cml==True:
        with open(dir_path_current.joinpath(
            'ac_par_il_' + length_name + '.pkl'
            ), 'wb') as f:
            pkl.dump(globals()['ac_par_il_' + length_name], f)
        f.close()

for ic, c in enumerate(cml_cent_sim):
    print('ITERATION %i' %ic)
    x = []
    y = []
    for i in range(len(x1)):
        x_temp = x1[i][km_in_start:km_in_end]
        y_temp = y1[i][km_in_start:km_in_end]
        x.append(x_temp)
        y.append(y_temp)
    x = np.array(x)
    y = np.array(y)
    bools = np.random.uniform(0,1,len(x.ravel())) > (1-links_density)
    x_links_cent = np.compress(bools,x)
    y_links_cent = np.compress(bools,y)

    num_of_links = len(y_links_cent)
    print('NUM OF LINKS IS %i' %num_of_links)

    x_links_cent = x_links_cent + np.random.normal(
        loc=0.0,
        scale=10.2,
        size=num_of_links
        )
    y_links_cent = y_links_cent + np.random.normal(
        loc=0.0,
        scale=10.2,
        size=num_of_links
        )

    links_cent = np.column_stack((x_links_cent,y_links_cent))

    cml_ids = link_id_gen(num_of_ids=num_of_links)

    for il, l in enumerate(cml_lengths):
        print('LENGTHS: %.2f' %l)
        ang = np.radians(
            np.random.uniform(low=0.0, high=179.9999, size=[num_of_links,1])
            )
        ang = np.squeeze(ang)
        links_lengths = l * 1e3 #np.ones(shape=[num_of_links,1])
        if identical_l is False:
            lower, upper, = (l-lag)*1e3, (l+lag)*1e3
            if l_dist == 'E':
                links_mean_length = l * 1e3
                X = stats.truncexpon(b=(upper-lower)/links_mean_length, 
                                    loc=lower, 
                                    scale=links_mean_length)
                links_lengths = X.rvs(num_of_links)
            elif l_dist == 'U':
                links_lengths = np.random.uniform(lower,upper,num_of_links)

        links_xa = (-links_lengths/2)*np.cos(ang) + links_cent[:,0]
        links_ya = (-links_lengths/2)*np.sin(ang) + links_cent[:,1]
        links_xb = (links_lengths/2)*np.cos(ang) + links_cent[:,0] 
        links_yb = (links_lengths/2)*np.sin(ang) + links_cent[:,1]

        lons_a, lats_a = transform(
            proj_meters, 
            proj_degrees, 
            links_xa, 
            links_ya
            )
        lons_b, lats_b = transform(
            proj_meters, 
            proj_degrees, 
            links_xb, 
            links_yb
            )

        df_sim_input = pd.DataFrame(
            columns= ['cml_id',
                'site_a_longitude',
                'site_b_longitude',
                'site_a_latitude',
                'site_b_latitude',
                'Frequency',
                'Length',
                'Polarization',
                'a','b','time','R_radolan','A']
            )

        df_sim_input['cml_id'] = cml_ids

        df_sim_input['Frequency'] = 23
        df_sim_input['site_a_longitude'], df_sim_input['site_a_latitude'] = \
                                transform(proj_meters, 
                                 proj_degrees, 
                                 links_xa, 
                                 links_ya)
        df_sim_input['site_b_longitude'], df_sim_input['site_b_latitude'] = \
                                transform(proj_meters, 
                                 proj_degrees, 
                                 links_xb, 
                                 links_yb)
        df_sim_input['Polarization'] = np.random.choice(
            ['H', 'V', 'V'], 
            df_sim_input.shape[0]
            )
        df_sim_input['Length'] = links_lengths / 1e3 # km

        for i, cml in df_sim_input.iterrows():
            df_sim_input.loc[i,'a'], df_sim_input.loc[i,'b'] = \
                pycml.processing.A_R_relation.A_R_relation.a_b(
                    cml['Frequency'],
                    cml['Polarization']
                    )

        df_sim_input.drop(['time'],axis='columns',inplace=True)
        # df_sim_input = df_sim_input.drop_vars('time')

        d_weights = {}
        for j, cml in enumerate(df_sim_input.cml_id.values): 
            intersec_weights = pycml.validation.validator.calc_intersect_weights(
                        x1_line=df_sim_input.site_a_longitude.values[j],
                        y1_line=df_sim_input.site_a_latitude.values[j],
                        x2_line=df_sim_input.site_b_longitude.values[j],
                        y2_line=df_sim_input.site_b_latitude.values[j],
                        x_grid=ds_radolan_GT.lon_grid.values,
                        y_grid=ds_radolan_GT.lat_grid.values,
                        grid_point_location='center')
            d_weights[cml] = intersec_weights


        list_of_GT_datasets = []
        list_of_radolan_along_cml = []
        QUANT = 'with'
        NOISE = 'with'

        for at, agg in enumerate(agg_times):
            # dir_path_current = dir_path.joinpath(str(agg_times[at]) + '_ts' +\
            #                               str(ts) + '_cod' + str(int(cod)) +\
            #                              'opt' + str(opt))
            # try:
            #     os.mkdir(dir_path_current)
            # except:
            #     nothing = 0
            print(str("ds_radolan_GT_" + agg))
            num_of_mins = float(split_at(agg,'T',1)[0])
            if aggregation_mean == True:
                globals()["ds_radolan_GT_" + agg] = ds_radolan_GT.resample(
                    time=agg, label='right', 
                    restore_coord_dims=False).mean(dim='time')
            else:
                globals()["ds_radolan_GT_" + agg] = ds_radolan_GT.resample(
                    time=agg, label='right', 
                    restore_coord_dims=False).sum(dim='time')
            list_of_GT_datasets.append(str("ds_radolan_GT_" + agg))

            path_ave_time = np.zeros(
                (len(df_sim_input.cml_id)))
            for j, cml in enumerate(df_sim_input.cml_id.values):
                path_ave_time[j] = round(
                    np.nansum(
                        d_weights[cml] * globals()["ds_radolan_GT_" + agg].\
                               raindepth.isel(time=ts).values), 
                                                6
                        )# * intensity_factor)

            df_sim_input['R_radolan'] = path_ave_time
            ## Discard zeros from cmls not yet working
            # if discard_zeros is True:
            #     df_sim_input = df_sim_input.where(df_sim_input['R_radolan'] < 0.0001)
            #     df_sim_input = df_sim_input[df_sim_input['R_radolan'].notna()]
            #     df_sim_input.reset_index(inplace=True,drop=True)

        df_sim_input.rename(columns = {'cml_id':'Link_num',
                                              'R_radolan':'R',
                                              'Length':'L',
                                              'Frequency':'F'}, 
                                   inplace=True)
        df_sim_input = cml_lat_lon_to_UTM(df_sim_input)
        df_sim_input, _ = gmz.create_virtual_gauges(df_sim_input, 
                                        num_gauges=1)

        ac = accml.Autocorr(df_sim_input, bw=bandwidth, cutoff_distance_km=cod)
        ac(optimize=opt)

        print('\n\n')
        length_name = round(l,1)
        length_name = split_at(str(format(length_name/100, '.3f')),'.',1)[-1]
        if save_cml==True:
            if ac.alpha_L * ac.beta_L <= 0:
                with open(dir_path_current.joinpath(
                        'ac_par_il_' + length_name + '.pkl'), 
                        'rb') as f:
                    arr = pkl.load(f)
                f.close()
                arr = np.row_stack((arr,np.array([666,666])))
                with open(dir_path_current.joinpath(
                        'ac_par_il_' + length_name + '.pkl'), 
                        'wb') as f:
                    pkl.dump(arr, f)
                f.close()
            else:
                with open(dir_path_current.joinpath(
                        'ac_par_il_' + length_name + '.pkl'), 
                        'rb') as f:
                    arr = pkl.load(f)
                f.close()
                arr = np.row_stack((arr,np.array( [ac.alpha_L, ac.beta_L, ac.gamma_L] )))
                with open(dir_path_current.joinpath(
                        'ac_par_il_' + length_name + '.pkl'), 
                        'wb') as f:
                    pkl.dump(arr, f)
                f.close()
###########################################
###########################################
## Save the respective radar dir path in the current directory
f = open(dir_path_current.joinpath('rad_dir.txt'), "w+")
f.write(str(rad_path_current))# + "\r\n")
f.close()

if analyze_radar == True:
    for _, agg in enumerate(agg_times):
        ## Radar ground truth empirical autocorrelation function
        data = globals()["ds_radolan_GT_" + agg].raindepth.isel(
            time=ts
            ).values.ravel().copy()
        print(len(data))
        temp_nans = np.argwhere(np.isnan(data))
        data = np.delete(data, temp_nans)
        print(len(data))

        ## grid definition for output field
        gridx = np.delete(
            globals()["ds_radolan_GT_" + agg].x_utm.values.ravel(), temp_nans
            )
        gridy = np.delete(
            globals()["ds_radolan_GT_" + agg].y_utm.values.ravel(), temp_nans
            )
        #######################
        ### Excluding zeros ###
        #######################
        if discard_zeros is True:
            bool_data = np.array(data,dtype=bool)
            data = np.compress(bool_data,data)
            gridx = np.compress(bool_data,gridx)
            gridy = np.compress(bool_data,gridy)
            print(len(data))
        #########################
        ######### Done ##########
        #########################
        def make_tuple_arr(arr):
            d_tuple = []
            for _,d in enumerate(arr):
                d_tuple.append(tuple((d,)))
            return d_tuple

        data = make_tuple_arr(data)
        gridx = make_tuple_arr(gridx)
        gridy = make_tuple_arr(gridy)

        df = pd.DataFrame({'x': gridx,
                        'y': gridy,
                        'z': data})
        print(dir_path_current)
        ac = accml.Autocorr(df, bw=bandwidth, cutoff_distance_km=cod)
        ac(optimize=opt)
        print(ac.alpha_L,ac.beta_L, ac.gamma_L)

        ## Pickle radar data
        df.to_pickle(rad_path_current.joinpath('df_radar.pkl'))

        with open(rad_path_current.joinpath('rad'+agg+'_acf.pkl'), 'wb') as f:
            pkl.dump(ac.ac[1], f)
        f.close()

        with open(rad_path_current.joinpath('rad'+agg+'_alpha.pkl'), 'wb') as f:
            pkl.dump(ac.alpha_L, f)
        f.close()

        with open(rad_path_current.joinpath('rad'+agg+'_beta.pkl'), 'wb') as f:
            pkl.dump(ac.beta_L, f)
        f.close()

        with open(rad_path_current.joinpath('rad'+agg+'_gamma.pkl'), 'wb') as f:
            pkl.dump(ac.gamma_L, f)
        f.close()

        with open(rad_path_current.joinpath('rad'+agg+'_nugget.pkl'), 'wb') as f:
            pkl.dump(ac.nugget, f)
        f.close()

        with open(rad_path_current.joinpath('rad'+agg+'_hs.pkl'), 'wb') as f:
            pkl.dump(ac.hs, f)
        f.close()

        # dat = [ac.alpha_L, ac.beta_L, ac.nugget, ac.hs]
        # with open(dir_path_current.joinpath(
        #     'RADAR'+ agg +'_alpha_beta_nugget_hs.dat'), 'wb') as f:
        #     pkl.dump(len(dat), f)
        #     for var in dat:
        #         pkl.dump(var, f)
        # f.close()
else:
    print('Radar autocorr has already been calcualted.')
    print(dir_path_current)

