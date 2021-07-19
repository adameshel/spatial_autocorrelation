import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import quad, nquad, dblquad
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime
from pathlib import Path
import pickle as pkl
import os
import shutil
from autocorr_functions import *

my_path = Path('/home/adameshel/Documents/code/autocorr/'+\
    'TRY/')
try:
    os.makedirs(my_path)
except:
    print('Directory %s already exsists' %str(my_path))

full_expr = False # Full or simplified expression
printing = False
## Define functions describing the parameters of exp.
alpha_beta_95 = False # When false alpha and beta are optimized

pic = 0
eps = 0.0
total_time = datetime.datetime.now() - datetime.datetime.now()

## Define parameters
thetas = np.linspace(-np.pi/2+eps, np.pi/2-eps, 5)
phis = np.linspace(-np.pi/2+eps, np.pi/2-eps, 5)
Ls = np.linspace(0.1,7,5)
set_of_alphas = np.linspace(5.0,20,3)
set_of_betas = np.array([2.5])#np.linspace(1.0,10,7)
# set_of_gammas = np.array([0.5,0.8,1.1,1.3,1.6])
gamma = 1.0
# set_of_alphas = np.arange(22.0,27.0,1.1)
# set_of_betas = np.arange(22.0,27.0,1.1)
hs = np.linspace(0.01,70.0,80)

a_all = np.zeros((len(set_of_alphas),len(set_of_betas)))
b_all = np.zeros((len(set_of_alphas),len(set_of_betas)))
r2a_all = np.zeros((len(set_of_alphas),len(set_of_betas)))
r2b_all = np.zeros((len(set_of_alphas),len(set_of_betas)))
gamma_list_of_lists = []

loc_j = 0

for alpha in set_of_alphas:
    loc_i = 0
    for beta in set_of_betas:
        Rr_mat = np.zeros((len(Ls),len(hs)))
        # total_time = datetime.datetime.now() - datetime.datetime.now()
        for i,L in enumerate(Ls):
            then = datetime.datetime.now()
            print(L)
            Rr_mat[i,:] = compute_acf(
                hs, 
                L, 
                alpha, 
                beta, 
                gamma, 
                thetas, 
                phis, 
                full_expr=full_expr
                )
            now = datetime.datetime.now()
            diff = now-then
            print('diff=%s' % diff)
            total_time += diff
            print('total=%s' % total_time)
            print('\n')

        ## Pickle data
        data = [Ls, hs, Rr_mat, alpha, beta]
        with open(my_path.joinpath(
            'Ls_hs_Rrmat_a_%ib_%i' %(
                alpha*100,beta*100
                ) + str(full_expr) + '.dat'
        ), 'wb') as f:
            pkl.dump(len(data), f)
            for var in data:
                pkl.dump(var, f)
        f.close()
        hc_l = [] # correlation distance list
        s_l = [] # sill list
        g_l = [] # gamma list
        for i in range(len(Rr_mat[:,0])):
            if alpha_beta_95:
                # Max value of ACF
                s_l.append(Rr_mat[i][0])
                # Correlatio distance as the h where 
                #the value of ACF decreases in 95%
                hc_l.append(
                    hs[np.sum(Rr_mat[i] >= np.nanmax(Rr_mat[i]) * 0.05) -1]
                    )
                g_l.append(1.0)
            else:
                try:
                    popt, _ = curve_fit(
                        f=acf_original, 
                        xdata=hs, 
                        ydata=Rr_mat[i],
                        p0=[1,1,1],
                        bounds=[0,(1e4,np.inf,7)]
                        )
                    alpha_opt, beta_opt, gamma_opt = popt
                except:
                    print(Rr_mat[i])
                    print('Optimization not converging')
                    alpha_opt = np.nan
                    beta_opt = np.nan
                    gamma_opt = np.nan
                s_l.append(beta_opt)
                hc_l.append(alpha_opt)
                g_l.append(gamma_opt)

        betas = np.array(s_l)
        alphas = np.array(hc_l)
        gammas = np.array(g_l)
        gamma_list_of_lists.append(g_l)
        ##############################
        ######## exclude nans ########
        ##############################
        alphas_nn, betas_nn, gammas_nn, Ls_nn = exclude_nans(
            alphas, 
            betas, 
            gammas, 
            Ls
            )
        ##############################
        ####### done with nans #######
        ##############################
        try:
            palpha, _ = curve_fit(
                f=alpha_L,
                xdata=np.array([Ls_nn,gammas_nn]),
                ydata=alphas_nn
                )

            a, alpha_0 = palpha
            line = alpha_L(
                np.array([Ls_nn,gammas_nn]), 
                a, 
                alpha_0
                )
            r2a = round(np.corrcoef(alphas_nn,line)[0,1],3)**2
            if r2a < 0.7:
                print('in alpha\n')
                print(alpha,beta,L)
            
            epsilon = 1e-7
            pbeta, _ = curve_fit(
                f=beta_L,
                xdata=Ls_nn,
                ydata=betas_nn,
                p0=[3,1,alpha_0],
                bounds=[(0,0,(alpha_0-epsilon)), (10, np.inf, (alpha_0+epsilon))]
                )
            b, beta_0, aa = pbeta
            print('alpha_0 is %.3f and aa is %.3f' %(alpha_0,aa))
            line = beta_L(Ls_nn, b, beta_0, alpha_0)
            r2b = round(np.corrcoef(betas_nn,line)[0,1],3)**2
            if r2b < 0.7:
                print('in beta\n')
                print(alpha,beta,L)
        except:
            print(Ls_nn)
            print(alphas_nn)
            a = np.nan; alpha_0 = np.nan; r2a = np.nan
            b = np.nan; beta_0 = np.nan; r2b = np.nan
        a_all[loc_j,loc_i] = a
        b_all[loc_j,loc_i] = b
        r2a_all[loc_j,loc_i] = r2a
        r2b_all[loc_j,loc_i] = r2b
        loc_i += 1
    loc_j += 1

## Pickle data
data = [set_of_alphas, set_of_betas, a_all, 
b_all, r2a_all, r2b_all, Ls, gamma_list_of_lists]
with open(my_path.joinpath(
    'as_and_bs' + str(full_expr) + '.dat'
    ), 'wb') as f:
    pkl.dump(len(data), f)
    for var in data:
        pkl.dump(var, f)
f.close()
print(my_path)