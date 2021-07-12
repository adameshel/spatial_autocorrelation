import numpy as np
import matplotlib.pyplot as plt

def acf_original(h, alpha, beta, gamma=1.0):
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
    return beta * np.exp((-L * np.sqrt(sq)) / alpha)


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
    c = np.linspace(0.0,0.8,int(len(hs)/3)+1)
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
    g_l = [] # gamma list
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
            g_l.append(1.0)
        else:

            popt, _ = curve_fit(f=acf_original, 
                                   xdata=hs, 
                                   ydata=Rr_mat[i])
            alpha_opt, beta_opt, gamma_opt = popt

            s_l.append(beta_opt)
            hc_l.append(alpha_opt)
            g_l.append(gamma_opt)
    betas = np.array(s_l)
    alphas = np.array(hc_l)
    gammas = np.array(g_l)
    return alphas, betas, gammas


def bias(x, b):
    '''bias term in linear fit'''
    return x + b


def combine_legend_subplots(i,xy=(2.8,1.01),fs=12):
    '''
    fs : font size
    '''
    handles, labels = ax[i].get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    lgd = ax[i].legend(handles, labels, loc='best', bbox_to_anchor=xy, fontsize=fs)
    return lgd