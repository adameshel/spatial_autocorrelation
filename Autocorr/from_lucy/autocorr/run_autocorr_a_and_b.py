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

my_path = Path('/home/adameshel/Documents/code/autocorr/with_gamma5/')
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

############################################################
###### put autocorr_functions.py here out of laziness ######
############################################################

def acf_original(h, alpha, beta):
    return beta * np.exp(-(h/alpha))

def acf_original_gamma(h, alpha, beta, gamma=1.0):
    return beta * np.exp(-(h/alpha)**gamma)

def acf(t1,t2,h,L,theta,phi=-999):
    '''Auto-Correlation Function'''
    # the full expression under the sqrt
    if phi != -999: 
        sq = t2**2 + (h/L)**2 + 2*t2*(h/L)*np.cos(theta-phi) +\
        t1**2 -2*t1*(t2*np.sin(theta)+(h/L)*np.sin(phi)) + 1e-8
    # the short expression where theta=phi
    else: 
        sq = (t2+(h/L))**2 + t1**2 -2*t1*(t2+(h/L))*np.sin(theta) + 1e-8
    if sq < 0:
        print(sq, theta, phi)
    return beta * np.exp(-((L * np.sqrt(sq)) / alpha))

def Rr_ang(h,L,thetas,phis,full_expr=False):
    '''Function that loops over thetas and returns 
    the solution to the double integral of acf, for the simplified expression,
    where theta equals phi (False) and for the full one (True).'''
    rs = []
    if full_expr is False:
        phis = np.array([-999])  
    for theta in thetas:
        r = []
        for phi in phis:    
            # perform the double integral
            result, err = nquad(acf,[[-0.5,0.5],[-0.5,0.5]],args=(h,L,theta,phi))#,
#                                 opts=[{'epsabs' : 1.49e-1,
#                                        'epsrel' : 1.49e-1,
#                                        'limit' : 2},
#                                       {'epsabs' : 1.49e-1,
#                                        'epsrel' : 1.49e-1,
#                                        'limit' : 2},
#                                       {},{}])
            r.append(result)
        r = np.array(r)
        rs.append(r)
    rs = np.array(rs)
    return rs, np.nanmean(rs) 

def compute_acf(hs,L,pic=0,printing=False):
    avgs = []
    i = 0; j = 0
    c = np.linspace(0.0,1.2,int(len(hs)/3)+1)
    for h in hs:
        acf_theta, avg = Rr_ang(h,L,thetas,phis,full_expr=full_expr) 
        avgs.append(avg)
        if printing and i%3==0:
            temp = h/L
            col = (c[j]/1.5, c[j]/1.5, c[j]/1.5)
            j += 1
            ax[pic].plot(thetas, acf_theta, color=col, label='h/L=%.2f' %temp)
        i += 1
    return np.array(avgs)

def alpha_L(x,a,alpha_0):
    return a*x + alpha_0
#     return a*x + alpha_0/x
#     return alpha_0*np.exp(x/(a*alpha_0))

def beta_L(x,b,beta_0):
    return beta_0*np.exp(-x/(b*alpha_0))
#     return alpha_0*b*x + beta_0

def gamma_L(x,c,gamma_0):
    return x ** c + gamma_0

def alpha_L_inv(x,a,alpha,bias=0):
    alpha_0 = alpha - a*x - bias
    return alpha_0

def beta_L_inv(x,b,beta,alpha_0):
    beta_0 = beta / (np.exp(-x/(b*alpha_0)))
#     beta_0 = beta / (alpha_0*b*x)
    return beta_0

def alpha_beta(hs, Rr_mat, Ls, alpha_beta_95=False, printing=False):
    hc_l = [] # correlation distance list
    s_l = [] # sill list
    c = np.linspace(0.0,1.0,len(Rr_mat[:,0])+1)
    for i in range(len(Rr_mat[:,0])):
        if printing:
            gs = np.array([c[i]/1.5, c[i]/1.5, c[i]/1.5])
            l = 'L=%.2f' %Ls[i]
            ax[0].plot(hs,Rr_mat[i],color=gs,label=l)
            ax[0].legend()
        if alpha_beta_95:
            # Max value of ACF
            s_l.append(Rr_mat[i][0])
            # Correlatio distance as the h where 
            #the value of ACF decreases in 95%
            hc_l.append(hs[np.sum(Rr_mat[i] >= np.nanmax(Rr_mat[i]) * 0.05) -1])
        else:
            popt, _ = curve_fit(f=acf_original, 
                                   xdata=hs, 
                                   ydata=Rr_mat[i])
            alpha_opt, beta_opt = popt

            s_l.append(beta_opt)
            hc_l.append(alpha_opt)
    betas = np.array(s_l)
    alphas = np.array(hc_l)
    return alphas, betas

def alpha_beta_gamma(hs, Rr_mat, Ls, printing=False):
    hc_l = [] # correlation distance list
    s_l = [] # sill list
    g_l = [] # gamma list
    c = np.linspace(0.0,1.0,len(Rr_mat[:,0])+1)
    for i in range(len(Rr_mat[:,0])):
        if printing:
            gs = np.array([c[i]/1.5, c[i]/1.5, c[i]/1.5])
            l = 'L=%.2f' %Ls[i]
            ax[0].plot(hs,Rr_mat[i],color=gs,label=l)
            ax[0].legend()
        popt, _ = curve_fit(
            f=acf_original_gamma, 
            xdata=hs, 
            ydata=Rr_mat[i],
            p0=[1,1,1],
            bounds=[0,(1e4,np.inf,7)]
        )
        alpha_opt, beta_opt, gamma_opt = popt

        s_l.append(beta_opt)
        hc_l.append(alpha_opt)
        g_l.append(gamma_opt)
    betas = np.array(s_l)
    alphas = np.array(hc_l)
    gammas = np.array(g_l)
    return alphas, betas, gammas

############################################################
#################### Done with laziness ####################
############################################################

## Define parameters
thetas = np.linspace(-np.pi/2+eps, np.pi/2-eps, 15)
phis = np.linspace(-np.pi/2+eps, np.pi/2-eps, 15)
Ls = np.linspace(0.1,25,15)
# set_of_alphas = np.linspace(0.1,25,20)
# set_of_betas = np.linspace(0.1,25,21)
set_of_alphas = np.arange(22.0,27.0,1.1)
set_of_betas = np.arange(22.0,27.0,1.1)
hs = np.linspace(0.01,40.0,60)

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
            Rr_mat[i,:] = compute_acf(hs, L)
            now = datetime.datetime.now()
            diff = now-then
            print('diff=%s' % diff)
            total_time += diff
            print('total=%s' % total_time)
            print('\n')

        ## Pickle data
        data = [Ls, hs, Rr_mat, alpha, beta]
        with open(my_path.joinpath(
            'Ls_hs_Rrmat_a_%ib_%i' %(alpha*100,beta*100) + str(full_expr) + '.dat'
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
                hc_l.append(hs[np.sum(Rr_mat[i] >= np.nanmax(Rr_mat[i]) * 0.05) -1])
                g_l.append(1.0)
            else:
                try:
                    popt, _ = curve_fit(
                        f=acf_original_gamma, 
                        xdata=hs, 
                        ydata=Rr_mat[i],
                        p0=[1,1,1],
                        bounds=[0,(1e4,np.inf,7)]
                        )
                    alpha_opt, beta_opt, gamma_opt = popt
                except:
                    print(Rr_mat[i])
                    print('Optimization not converging')
                    alpha_opt = np.nan#hs[np.sum(Rr_mat[i] >= np.nanmax(Rr_mat[i]) * 0.05) -1] / 3.0
                    beta_opt = np.nan#Rr_mat[i][0] 
                    gamma_opt = np.nan
                s_l.append(beta_opt)
                hc_l.append(alpha_opt)
                g_l.append(gamma_opt)

        betas = np.array(s_l)
        alphas = np.array(hc_l)
        gamma_list_of_lists.append(g_l)
        ##############################
        ######## exclude nans ########
        ##############################
        bool_data = ~np.isnan(alphas,dtype=bool)
        alphas_nn = np.compress(bool_data,alphas)
        betas_nn = np.compress(bool_data,betas)
        Ls_nn = np.compress(bool_data,Ls)
        ##############################
        ####### done with nans #######
        ##############################
        try:
            palpha, _ = curve_fit(f=alpha_L,xdata=Ls_nn,ydata=alphas_nn)
        except:
            print(Ls_nn)
            print(alphas_nn)
        a, alpha_0 = palpha
        line = alpha_L(Ls_nn, a, alpha_0)
        r2a = round(np.corrcoef(alphas_nn,line)[0,1],3)**2
        if r2a < 0.7:
            print('in alpha\n')
            print(alpha,beta,L)

        pbeta, _ = curve_fit(f=beta_L,xdata=Ls_nn,ydata=betas_nn)
        b, beta_0 = pbeta
        line = beta_L(Ls_nn, b, beta_0)
        r2b = round(np.corrcoef(betas_nn,line)[0,1],3)**2
        if r2b < 0.7:
            print('in beta\n')
            print(alpha,beta,L)
        a_all[loc_j,loc_i] = a
        b_all[loc_j,loc_i] = b
        r2a_all[loc_j,loc_i] = r2a
        r2b_all[loc_j,loc_i] = r2b
        loc_i += 1
    loc_j += 1

## Pickle data
data = [set_of_alphas, set_of_betas, a_all, b_all, r2a_all, r2b_all, Ls, gamma_list_of_lists]
with open(my_path.joinpath('as_and_bs' + str(full_expr) + '.dat'), 'wb') as f:
    pkl.dump(len(data), f)
    for var in data:
        pkl.dump(var, f)
f.close()
print(my_path)