CODE_DIR = r'C:/Users/Matteo/Documents/GitHub/netlds/'
RESULTS_DIR_BASE = r'C:/Users/Matteo/Documents/UNI/Columbia/Rotation_Liam/'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as clr
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import numpy.random as rnd
import scipy.special as spc
import scipy.stats as sts
import scipy.sparse as spr
import pandas as pd
import pandas.plotting as pdplt
from sklearn.neighbors import NearestNeighbors as knn

import time
import sys
import os
import glob
import warnings
sys.path.insert(0, CODE_DIR)
from netlds.models import *
from netlds.generative import *
from netlds.inference import *
from data.sim_data import build_model
from tqdm import tqdm

warnings.filterwarnings("ignore", 
                        message="using a non-tuple sequence for multidimensional indexing is deprecated")

#%% function for initialising GMM
def initialise(X, priors, how = 'prior'):
    '''
    X (n_obs, dim_x)
    priors = [pi, mu, sig]
    how = 'prior' or 'randkm'
    '''
    
    pi, mu, sig = priors
    n_obs, n_dim = X.shape
    K = pi.shape[0]
    
    s = 0.5 # how much to perturb
    
    # Initialise the values
    if how == 'prior': # perturbation from prior
        mu_init = mu*(1 - s + 2*s*rnd.rand(1,K))
        sig_init = sig*(1 - s + 2*s*rnd.rand(1,1,K)) 
        pi_init = pi*(1 - s + 2*s*rnd.rand(K,1))
        pi_init /= pi_init.sum(0)[:,np.newaxis].T
        
        distX = np.linalg.norm(X[:,:,np.newaxis] - mu_init[np.newaxis,:,:],axis = 1)
        z_init = np.zeros((n_obs,K))
        z_init[np.arange(n_obs),np.argmin(distX, axis = 1)] = 1 
        
    elif how == 'randkm': # sort of k-means thing
        s = 0.1
#        if n_dim >= K:
        d = rnd.permutation(np.arange(n_dim))
#        else:
#            d = np.append(rnd.permutation(np.arange(n_dim)),
#                          rnd.choice(n_dim, K-n_dim))
        whichdims = [d[0+n::K] for n in np.arange(K)]

        q = np.quantile(X, [k*(1./K) for k in range(K+1)], axis = 0)
        mean_qk = np.zeros((n_dim,K))
        for k in range(1,K+1): # get quantile means
            in_qk = [(X[:,d] < q[k,d]) & (X[:,d] >= q[k-1,d]) for d in range(n_dim)]
            mean_qk[:,k-1] = [X[in_qk[d],d].mean(0) for d in range(n_dim)]
            
        mu_init = np.zeros((n_dim,K))
        for k in range(K): # select class means
            trailing = [whichdims[dd] for dd in rnd.permutation(np.setdiff1d(range(K),k))]
            nt = len(trailing)
            mu_init[whichdims[k],k] = mean_qk[whichdims[k],-1]
            for kk in range(nt):
                mu_init[trailing[kk],k] = mean_qk[trailing[kk],-(kk+2)]
        mu_init *= (1 - s + 2*s*rnd.rand(1,K))

        distX = np.linalg.norm(X[:,:,np.newaxis] - mu_init[np.newaxis,:,:],axis = 1)
        z_init = np.zeros((n_obs,K))
        z_init[np.arange(n_obs),np.argmin(distX, axis = 1)] = 1
        
        pi_init = np.tile(z_init.sum(0)/n_obs, (n_obs,1)).T

        sig_init = np.array([np.cov(X.T, aweights = z_init[:,k]) for k in range(K)]).T

    pi_init = pi_init.astype(np.float32)
    mu_init = mu_init.astype(np.float32)
    sig_init = sig_init.astype(np.float32)

    return pi_init, mu_init, sig_init, z_init

#%% neighbourhood functions
def neighbour_func(S, w_i, nb_func = np.mean, nbin = 5, how = 'evenly'):
    '''
    compute a function of w_i for each point in S within a binned neighbourhood
    
    wrapper for 'scipy.stats.binned_statistic_dd' function
    
    S (N, dim_s)
    w_i (N,)
    nb_func callable: 1d --> scalars; default is np.mean
    nbin int; default 5
    how {'evenly', 'quantiles'}; default 'evenly'
    '''
    
    dim_s = S.shape[1]
    if how == 'evenly':
        bins = tuple([np.linspace(S[:,d].min(),S[:,d].max(),nbin+1) for d in range(dim_s)])
    elif how == 'quantiles':
        bins = tuple([np.quantile(S[:,d],np.arange(nbin+1)/nbin) for d in range(dim_s)])
        
    stat ,_, which_nb = sts.binned_statistic_dd(S, w_i, statistic = nb_func, bins = bins)
    
    if dim_s != 1:
        which_nb -= (nbin+3) # need to correct for silly indexing
        for b in range(nbin):
            deez = np.isin(which_nb,[range(b*(nbin+2),(b+1)*(nbin+2))])
            which_nb[deez] -= b*2
    else:
        which_nb -= 1
    
    pi_i = stat.flatten()[which_nb]
    
    return pi_i, which_nb
    
#%% helpers
def mixture_density_2d(lims,mus,covs,pies, num = 100):
    '''
    mus (n_dim,K)
    covs (n_dim,n_dim,K)
    pies (1,K)
    '''
    
    Q = np.array(np.meshgrid(np.linspace(lims[0,0],lims[0,1],num),np.linspace(lims[1,0],lims[1,1],num))).T
    
    probs = np.array([pies[k]*sts.multivariate_normal.pdf(Q,mus[:,k],covs[:,:,k]) for k in range(K)])
    probs = probs.sum(0)
    
    return (probs, Q)

def remove_outliers(X, k, q = 0.1):
    
    foo = knn(k + 1)
    foo.fit(X)
    dist = foo.kneighbors(X,return_distance = True)[0][:,1:]
    dens = 1/np.mean(dist,axis = 1)
    
    keepers = (dens >= np.quantile(dens, q))
    
    return X[keepers,:]

#%% plotting functions
def cov_ellipse(C, m, conf = 0.95):
    '''
    get ellipse of covariance C
    '''
    cf = sts.chi2.ppf(conf,2)

    L, V = np.linalg.eigh(C)
    order = L.argsort()[::-1]
    L, V = L[order], V[:, order]
    
    a = 2*np.sqrt(cf*L[0])
    b = 2*np.sqrt(cf*L[1])
    
    t = np.linspace(0,2*np.pi,100)
    tht = np.arctan2(V[1,0],V[0,0])
    
    x = m[0] + a*np.cos(t)*np.cos(tht) - b*np.sin(t)*np.sin(tht)
    y = m[1] + b*np.sin(t)*np.cos(tht) + a*np.cos(t)*np.sin(tht)
    
    return x, y

def soft_colormap(class_probs, cmap_name = 'jet', nbin = 200):
    
    N, K = class_probs.shape
    
    vals = np.argmax(class_probs, axis = 1)
    foo = cm.ScalarMappable(cmap = cmap_name)
    hsv = clr.rgb_to_hsv(foo.to_rgba(vals*(255/K))[:,:3])

    hsv[:,1] = (class_probs[range(N),vals] - (1/K))/(1 - (1/K))
    cols = clr.hsv_to_rgb(hsv)
    
    return cols

#%% load data
data_dirs = os.listdir(RESULTS_DIR_BASE + r'/data')
binsize = 0.2 # 200 ms bins
epoch = [4564,5501]
whichEpoch = 'epoch4'

bins = np.arange(epoch[0],epoch[1],binsize)

#st = np.load(r'C:\Users\Matteo\Documents\UNI\Columbia\Rotation_Liam\data\sorted.spike.time2.npy')
#clu = np.load(r'C:\Users\Matteo\Documents\UNI\Columbia\Rotation_Liam\data\sorted.spike.cluster2.npy')
#cids = np.unique(clu)
#Y = np.array([np.histogram(st[clu == c],bins = bins)[0] for c in cids])

# for loading the raw data (?)
nbin = bins.shape[0] - 1
Y = np.empty((0,nbin))
st = np.empty(0)
st_multi = np.empty(0)
clu = np.empty(0)
dset = np.empty(0)
mua_dset = np.empty(0)
amps = np.empty((0,4))
ii = 0
for dd in data_dirs:
    ii += 1
    if ('.mat' in dd) or ('.npy' in dd):
        continue
    
    trode = int(dd[:2])
    
    spks_tmp = np.load(RESULTS_DIR_BASE + r'/data/' + dd + r'\spike.times.npy')
    clus_tmp = np.load(RESULTS_DIR_BASE + r'/data/' + dd + r'\spike.cluster.npy')
    feat_tmp = np.load(RESULTS_DIR_BASE + r'/data/' + dd + r'\spike.features.npy').T
    
    goodclu = np.sum(np.isnan(feat_tmp),axis = 1) == 0
    good_mua = (clus_tmp[:,0] == 0) & goodclu
    if clu.size > 0:
        m = np.max(clu) 
    else:
        m = 0
    
    st_multi = np.append(st_multi,spks_tmp[good_mua]/10000)
    mua_dset = np.append(mua_dset,np.ones(np.sum(good_mua))*trode)    
    
    amps = np.append(amps,feat_tmp[good_mua,-4:],axis = 0)
    
    st = np.append(st,spks_tmp[clus_tmp > 0]/10000)
    clu = np.append(clu,clus_tmp[clus_tmp > 0] + m)
    dset = np.append(dset,np.ones(np.sum(clus_tmp > 0))*trode)
    cids = np.unique(clus_tmp[clus_tmp > 0])
    if cids.size > 0:
        sp = np.array([np.histogram(spks_tmp[clus_tmp == c]/10000,bins = bins)[0] for c in cids])
        Y = np.append(Y, sp, axis = 0)
        print(dd + ': done')

clu = clu[np.argsort(st)]  
dset = dset[np.argsort(st)]
mua_dset = mua_dset[np.argsort(st_multi)]
amps = amps[np.argsort(st_multi),:]
st = np.sort(st)
st_multi = np.sort(st_multi)

pos = np.load(r'C:\Users\Matteo\Documents\UNI\Columbia\Rotation_Liam\data\behave.position4.npy')
t_pos = np.load(r'C:\Users\Matteo\Documents\UNI\Columbia\Rotation_Liam\data\behave.time4.npy')

#%% define dataset
this_dset = 12
# data can be: [1,  2,  3,  4,  5,  7,  8, 10, 11, 12, 13, 14, 
#              17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29]
q = 0.1 # quantile cutoff
k = 20 # how many neighbours?

mua_epoch = st_multi[(st_multi > epoch[0])&(st_multi < epoch[1]) & (mua_dset == this_dset)]
x_multi = np.interp(mua_epoch, t_pos[:,0], pos[:,0])
y_multi = np.interp(mua_epoch, t_pos[:,0], pos[:,1])

X = amps[(st_multi > epoch[0])&(st_multi < epoch[1]) & (mua_dset == this_dset),:]
x_cntr = (X - X.mean(0)).T
S = np.array([x_multi, y_multi]).T
#S = x_multi[:,np.newaxis]

#foo = knn(k + 1)
#foo.fit(X)
#dist = foo.kneighbors(X,return_distance = True)[0][:,1:]
#dens = 1/np.mean(dist,axis = 1)
#keepers = (dens >= np.quantile(dens, q))
#
#X_full = X
#X = X[keepers,:]
#S_full = S
#S = S[keepers,:]

N = X.shape[0]
print('\nFitting for tetrode %s, which has %d spikes' % (this_dset, N))
#%% fit mixture model
max_iter = 50
n_runs = 200
K = 7
init_method = 'prior' # how to initialise, 'prior' or 'randkm'
bin_method = 'evenly' # how to bin S, 'evenly' or 'quantiles'
use_svgmm = False # allow proportions to vary per-datapoint?
nb_bins = 5

# initialise
mu_prior = X.mean(0)[:,np.newaxis]
sig_prior = (x_cntr.dot(x_cntr.T)[:,:,np.newaxis]/(N - 1))
pi_prior = (np.ones((K,N))/K)

# run
elbo = np.zeros((n_runs,max_iter))
pies = np.zeros((K,N,max_iter+1))
mus = np.zeros((X.shape[1],K,max_iter+1))
sigs = np.zeros((X.shape[1],X.shape[1],K,max_iter+1))
dubs = np.zeros((N,K,max_iter+1))
maxlik = -np.inf
for r in tqdm(range(n_runs)):
    pi, mu, sig, z_init = initialise(X, [pi_prior, mu_prior, sig_prior], how = init_method)
    
    pies[:,:,0] = pi # save initial conditions
    mus[:,:,0] = mu
    sigs[:,:,:,0] = sig
    dubs[:,:,0] = z_init
    
    for n in range(max_iter):
        # E step
        log_w = np.array([sts.multivariate_normal.logpdf(X,mu[:,k],sig[:,:,k]) for k in range(K)])
        log_w += np.log(pi + 1e-16)
        c = log_w.max(0)
        log_norm = c + np.log(np.sum(np.exp(log_w - c), axis = 0))
        log_w -= log_norm
        w_ik = (np.exp(log_w).T[:,np.newaxis,:])
        # M step
        if use_svgmm:
            pi = np.array([neighbour_func(S, w_ik[:,0,k], nbin = nb_bins, how = bin_method)[0] for k in range(K)])
        else:
            pi = np.tile(np.squeeze(w_ik.sum(0)/N),(N,1)).T
        mu = np.array([np.average(X, axis = 0, weights = w_ik[:,0,k]) for k in range(K)]).T
        sig = np.array([np.cov(X.T, aweights = w_ik[:,0,k]) for k in range(K)])
        sig = np.transpose(sig,(1,2,0))
        # likelihood
        probs = np.array([pi[k]*sts.multivariate_normal.pdf(X,mu[:,k],sig[:,:,k]) for k in range(K)])
        loglik = np.sum(np.log(probs.sum(0)))
        elbo[r,n] = loglik
        
        pies[:,:,n+1] = pi
        mus[:,:,n+1] = mu
        sigs[:,:,:,n+1] = sig
        dubs[:,:,n+1] = np.exp(log_w).T
        
#    nparam = use_svgmm*nbin*4 + K*(X.shape[1] + X.shape[1]**2)/2
    maxlik = np.max([maxlik, loglik])
    if loglik == maxlik: # hold on to the best model
        best_pi = pies
        best_mu = mus
        best_sig = sigs
        best_w = dubs
        
dis = np.argmax(elbo[:,-1])
plt.figure()
plt.plot(elbo.T, c = [0.5,0.5,0.5],linewidth = 1)
plt.plot(elbo[dis,:].T,'k-',linewidth = 2)
plt.ylabel('log-likelihood')

#%% plot
it = -1
cmap_scat = 'hsv'
cmap_pdf = 'cool'
#col = np.argmax(best_w, axis = 1)
#col = soft_colormap(this_w, cmap_scat)
_, col = neighbour_func(S_full, np.ones(X_full.shape[0]), nbin = nb_bins)
#col = 'r'
#col = S[:,1]
alpha = 0.5
conf = 1./3.
plot_density = False

this_mu = best_mu[:,:,it]
this_sig = best_sig[:,:,:,it]
this_w = best_w[:,:,it]
cnt = neighbour_func(S, np.ones(N), nbin = nb_bins, nb_func = 'count')[0]
this_pi = np.average(best_pi[:,:,it], axis = 1, weights = cnt/np.sum(cnt))

#this_mu = mu
#this_sig = sig
#this_w = z_init

df = pd.DataFrame(X_full, columns = ('Trode1','Trode2','Trode3','Trode4'))
axs = pdplt.scatter_matrix(df, s = 5, c = col, alpha = alpha, zorder = 2)
plt.set_cmap(cmap_scat)

for ii in range(4):
    for jj in range(4):
        if ii == jj:
            continue
        if plot_density:
            lims = np.array([axs[jj,ii].get_xlim(),axs[jj,ii].get_ylim()])    
            C = this_sig[[ii,jj],:,:][:,[ii,jj]]
            m = this_mu[[ii,jj],:]
            pdf, Q = mixture_density_2d(lims, m, C, this_pi)
            axs[jj,ii].pcolormesh(Q[:,:,0],Q[:,:,1], pdf, cmap = cmap_pdf, zorder = 1)
        else:
            for k in range(K):
                C = this_sig[[ii,jj],:,k][:,[ii,jj]]
                m = this_mu[[ii,jj],k]
                ellipse_x, ellipse_y = cov_ellipse(C,m, conf = conf)
                ell = axs[jj,ii].plot(ellipse_x,ellipse_y, 'k-', linewidth = 1)
            mew = axs[jj,ii].scatter(this_mu[ii,:],this_mu[jj,:], c = 'k', marker = 'd')
        

#def init():
#    ell.set_data(ellipse_x, ellipse_y)
#    mew.set_offsets()

#%%
ellipse_x, ellipse_y = cov_ellipse(sts.wishart.rvs(10,np.eye(2)/2),m, conf = 1./3.)
plt.plot(ellipse_x,ellipse_y, 'k-', linewidth = 1)
