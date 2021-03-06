#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Jun 7 12:33:17 2021
### Change
@author: adameshel
'''

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
from autocorr_functions import *

def excludenans(arr):
    bool_data = ~np.isnan(arr,dtype=bool)
    return np.compress(bool_data,arr)

class Autocorr():
    def __init__(self, df, bw=1.0, cutoff_distance_km=90.0):
        """
        df can be either a 2d array of regularly gridded data (will be processed 
        using Wiener-Khintchine theorem) or a pd.Dataframe (processed by 
        1/N*SUM(z_i * z_j)).
        df columns can be -
            Link_num, L, x, y, z
            Link_num: the CML's unique serial number
            L: Length of CML in km
            x: (tuple) Necessary. The x coordinate (UTM) of the 
                measurement point on CML[i]
            y: (tuple) Necessary. The y coordinate (UTM) of the 
                measurement point on CML[i]
            z: (tuple) Necessary. The rain intensity of the 
                measurement point on CML[i]
        
        To convert your X,Y,Z arrays to the correct form use:
 
        def make_tuple_arr(arr):
            d_tuple = []
            for _,d in enumerate(arr):
                d_tuple.append(tuple((d,)))
            return d_tuple
        z = make_tuple_arr(Z); x = make_tuple_arr(X); y = make_tuple_arr(Y)
        df = pd.DataFrame(
            {'x':x,
            'y':y,
            'z':z}
        )
        bw - Bandwidth in km.
        cutoff_distance_km - The maximal distance between measurement points accounted for.
        """
        if isinstance(df, pd.DataFrame): # df is a Dataframe
            self.WKT = False
            bw = bw * 1e3 # convert to meters
            cutoff_distance_km = cutoff_distance_km * 1e3 # convert to meters
            if 'L' in df:
                xMax = np.max([df.xa.max(), df.xb.max()])
                xMin = np.min([df.xa.min(), df.xb.min()])
                yMax = np.max([df.ya.max(), df.yb.max()])
                yMin = np.min([df.ya.min(), df.yb.min()])
            else:
                xMax = np.max(df.x.max())
                xMin = np.min(df.x.min())
                yMax = np.max(df.y.max())
                yMin = np.min(df.y.min())
                
            p_prep = np.array( df[['x','y','z']] )
            max_num_of_rg_in_row = len(p_prep[0,2])
            ## A loop for defining the VRGs per link where the number  
            ## of VRGs per link is not the same
            # if type(p_prep[0,2]) is tuple:
            #     for row in range(len(p_prep[:,0])):
            #         num_of_rg_in_row = len(p_prep[row,0])
            #         p_prep[row,2] = p_prep[row,2][:num_of_rg_in_row]
            # else:
            #     for row in range(len(p_prep[:,0])):
            #         num_of_rg_in_row = 1
            #         p_prep[row,2] = p_prep[row,2]

            for row in range(len(p_prep[:,0])):
                num_of_rg_in_row = len(p_prep[row,0])
                p_prep[row,2] = p_prep[row,2][:num_of_rg_in_row]
                    
                
            p = np.zeros([len(p_prep[:,0]) * max_num_of_rg_in_row,
                        len(p_prep[0,:])])
            link_num_prep = []
            link_L_prep = []
            p_row = 0
            element = 0
            for row in range(len(p_prep[:,0])):
                if row != 0:
                    p_row = p_row + element +1
                for col in range(len(p_prep[0,:])):
                    for element in range(len(p_prep[row,2])):
                        # import pdb; pdb.set_trace()
                        p[p_row+element,col] = p_prep[row,col][element]
            for cut_point in range(len(p[:,0])): 
                if p[cut_point,0] == 0: 
                    break
            if len(p[:,0]) == (cut_point + 1):
                p_filtered = p
            else:
                p_filtered = np.delete(p, 
                                    range(cut_point,len(p_prep[:,0]) *\
                                            max_num_of_rg_in_row), 
                                    0)
            if 'L' in df:
                for row in range(len(p_prep[:,0])):
                    for element in range(len(p_prep[row,2])):
                        link_num_prep.append(df['Link_num'][row])
                        link_L_prep.append(df['L'][row])

            self.df_p = pd.DataFrame(p_filtered, columns=['x','y','z'])
            if 'L' in self.df_p:
                self.df_p['l_num'] = link_num_prep
                self.df_p['L'] = link_L_prep

            self.hs = np.arange(bw, 
                        np.min([np.max([xMax-xMin,yMax-yMin]), 
                        cutoff_distance_km]), 
                        bw*2.0)
            p = self.df_p[['x','y','z']].values
            self.distances = squareform( pdist( p[:,:2] ) )
            self.ac = self._AC( p, self.hs, bw )
        else: # df is gridded data which will undergo FFT. bw is now a gridpoint length.
            self.res = bw
            self.Z = df
            lenx = np.shape(df)[1]
            x = np.linspace(-(lenx/2.0)*self.res,(lenx/2.0)*self.res,lenx)
            leny = np.shape(df)[0]
            y = np.linspace(-(leny/2.0)*self.res,(leny/2.0)*self.res,leny)
            self.X, self.Y = np.meshgrid(x,y)
            fft = np.fft.fftshift((np.fft.fft2(self.Z))) / len(self.Z.ravel())
            self.s = abs(np.fft.fftshift(np.fft.ifft2(fft*np.conjugate(fft))))
            self.s = self.s / np.nanmax(self.s)
            self.WKT = True # Wiener-Khintchine theorem


    def __call__(self, optimize=True, beta_to_one=False):
        '''
        Choose method by which you wish to find  the exponential and 
        multiplication parameters- alpha_L and beta_L.
        
        optimize=True: scipy optimize acf_original to data.
        optimize=False: beta_L is the max value and alpha_L is the 
        value of h at which the function lost 95% (not recommended).

        beta_to_one: (bool) Used only when df is a regularly gridded 2d array. 
        beta is not optimized and set to 1.
        '''
        if self.WKT == False:
            ## Choose the nugget before optimizing
            if len(self.hs) > 15:
                mins = int(len(self.ac[1])/6)
                self.nugget = np.nanmedian(np.sort(self.ac[1])[:mins])
            else:
                self.nugget = np.min(self.ac[1])

            if optimize==True:
                self.magnitude_beta = 10 ** (int(np.log10(np.var(self.ac[1]))))
                self.magnitude_alpha = 10 ** (int(np.log10(np.nanmean(self.hs))))
                self.ac[0] = self.ac[0] / self.magnitude_alpha
                self.ac[1] = self.ac[1] / self.magnitude_beta
                self.nugget = self.nugget / self.magnitude_beta
                try:
                    popt, pcov = curve_fit(
                        f=acf_original, 
                        xdata=self.ac[0], 
                        ydata=self.ac[1]-self.nugget,
                        p0=[1,1,1],
                        bounds=[0,(1e4,np.inf,7)]
                    )
                    self.std_error = np.sqrt(np.diag(pcov))
                    self.alpha_L, self.beta_L, self.gamma_L = popt
                    self.alpha_L  = self.alpha_L * self.magnitude_alpha
                    self.beta_L = self.beta_L * self.magnitude_beta
                    self.ac[0] = self.ac[0] * self.magnitude_alpha
                    self.ac[1] = self.ac[1] * self.magnitude_beta
                    self.nugget = self.nugget * self.magnitude_beta
                except:
                    self.alpha_L = np.nan
                    self.beta_L = np.nan
                    self.gamma_L = np.nan
                    self.nugget = np.nan
            else:
                # Max value of ACF
                self.beta_L = self.ac[1][0]
                # Correlation distance as h where 
                #the value of ACF decreases by 95%
                self.alpha_L = \
                    self.hs[np.sum((self.ac[1] - self.nugget) >= \
                        (np.nanmax(self.ac[1] - self.nugget) * 0.05)) -1]
        else:
            if beta_to_one==True:
                epsilon = 1e-7
            else:
                epsilon = 0.5
            self.magnitude_alpha = 10 ** (round(np.log10(np.nanmean(abs(self.X.ravel())))))
            xdata = np.vstack(
                (self.X.ravel() / self.magnitude_alpha, 
                self.Y.ravel() / self.magnitude_alpha)
                )
            self.pars, pcov = curve_fit(
                f=acf_original_2d,
                xdata=xdata,
                ydata=self.s.ravel(),
                p0=[1*self.res,1,1],
                bounds=[(0, 1-epsilon, 0.1), 
                (np.inf, 1+epsilon, 10)]
            )   
            self.std_error = np.sqrt(np.diag(pcov))
            self.std_error[0] = self.std_error[0] * self.magnitude_alpha
            self.pars[0] = self.pars[0] * self.magnitude_alpha

            self.ac_2d = np.reshape(
                acf_original_2d(xdata, self.pars[0], self.pars[1], self.pars[2]),
                np.shape(self.X)
                )
        
    def _ACh( self, P, h, bw ):
        '''
        Experimental autocorrelation function for a single lag
        '''
        Z = list()
        for i in range( self.distances.shape[0] ):
            sub = self.distances[i,i:]
            idxs = np.argsort(sub) + i
            sub = np.sort(sub)
            sub = np.where( sub >= h-bw,sub,np.nan )
            ncnt = np.sum( np.isnan(sub,dtype=bool) )
            sub = np.where( sub <= h+bw,sub,np.nan )
            sub = excludenans( sub )
            Z.append( P[i,2] * P[idxs[ncnt:len(sub)+ncnt],2] )
        Z = np.concatenate( Z )
        
        if len( Z )==0:
            return -1
        return np.sum( Z ) / ( len( Z ) )
    
    def _AC( self, P, hs, bw ):
        '''
        Experimental autocorrelation function for a collection of lags
        '''
        ac = list()
        for h in hs:
            ac.append( self._ACh( P, h, bw ) )
        ac = [ [ hs[i], ac[i] ] for i in range( len( hs ) ) if ac[i] != -1 ]
        return np.array( ac ).T