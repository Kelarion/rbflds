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
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as rob

import time
#import sys
#import os
#import glob
import warnings
#from tqdm import tqdm

warnings.filterwarnings("ignore", 
                        message="using a non-tuple sequence for multidimensional indexing is deprecated")

numpy2ri.activate()
r_mat_package = importr('Matrix', on_conflict = 'warn')

#%%
class MixtureModel(object):
    '''Generic class for handling multiple GMMs of spike features'''

    def __init__(self, dimensions, num_comp, verbose = False):
        '''Initialize the most general model parameters.
        
        dimensions (int or list of int): the number of data dimensions
        num_comp (int or list of int): the number of mixture components'''
        
        if type(dimensions) is int:
            dimensions = [dimensions]
            num_comp = [num_comp]
        
        if len(dimensions) != len(num_comp):
            raise ValueError('dimensions and num_comp must be the same length')
        
        self.dim_obs = dimensions
        self.num_comp = num_comp
        self.num_dset = len(self.dim_obs)
        
        self.verbose = verbose
        
        # Nested list to store model parameters
        self.params = [[] for _ in range(self.num_dset)]
        
    #%% fitting
    def fit_model(
            self, data, max_iter = 200, n_runs = 10, init_method = 'prior', 
            use_spatial = False, num_spatial_bins = 5, bin_method = 'evenly',
            spatial_variable = None, p_remove = 0.1, skip_datasets = [],
            make_plots = True):
        '''
        Fit the mixture model using the standard EM algorithm. Automatically 
        removes outlier points.
        
        Args:
            data (list of (n_obs, dim_obs) nparray): list of datasets
            max_iter (int): maximum number of iterations in each run
            n_runs (int): number of times to run the algorithm
            init_method ('prior' or 'randkm'): initialisation method
            use_spatial (bool): fit the SVGMM?
            num_spatial_bins (int): how many spatial bins along each dimension
            bin_method ('evenly'): how to space the edges of each spatial bin
            spatial_variable (list of (n_obs,dim_spat) nparrays): the variable
                to use for the SVGMM
            p_remove (0-1): proportion of outliers to remove
            skip_datasets (list of int): list of any dataset indices to skip
            make_plots (bool): whether to show the log-likelihood plots
        '''
        
        if use_spatial and spatial_variable is None:
            raise ValueError('You need to provide a spatial variable if use_spatial is True!')
        
        num_dsets = len(data) - len(skip_datasets)
        
        # Fit each dataset
        t0 = time.time()
        dset = -1
        for dset, dat in enumerate(data):
            if dset in skip_datasets:
                continue
            
            num_clu = self.num_comp[dset]
            
            if self.verbose:
                print('%.1f: Fitting model %d/%d' \
                          % (time.time() - t0, dset+1, num_dsets))
                
            # remove outlier points
            nonoutlier = self.remove_outliers(dat, q = p_remove)
            dat = dat[nonoutlier,:]
            if spatial_variable is not None:
                SV = spatial_variable[dset][nonoutlier,:]
            
            N, d = dat.shape # number of data points
            
            # initialise
            mu_prior = dat.mean(0)[:,np.newaxis]
            sig_prior = (dat.T.dot(dat)[:,:,np.newaxis]/(N - 1))
            pi_prior = (np.ones((num_clu, N))/num_clu)
            
            # run the EM
            elbo = np.zeros((n_runs, max_iter))
#            pies = np.zeros((num_clu, N, max_iter+1))
#            mus = np.zeros((dat.shape[1], num_clu, max_iter+1))
#            sigs = np.zeros((dat.shape[1], dat.shape[1], num_clu, max_iter+1))
#            dubs = np.zeros((N, num_clu, max_iter+1))
            maxlik = -np.inf
            for r in range(n_runs):
                self.initialise_model(dset, dat, [mu_prior, sig_prior, pi_prior], 
                                      how = init_method, s = 0.1)
                
                # should I save initial conditions?
#                pies[:,:,0] = pi 
#                mus[:,:,0] = mu
#                sigs[:,:,:,0] = sig
#                dubs[:,:,0] = z_init
                mu, sig, pi, _ = self.params[dset]
                
                for n in range(max_iter):
                    # E step
                    self.E_Step(dset, dat)
                    
                    # M step
                    if use_spatial:
                        self.M_Step_SV(dset,dat,SV,num_spatial_bins, bin_method)
                    else:
                        self.M_Step(dset,dat)
                    
                    # likelihood
                    probs = self.pdf(dset,dat)[0]
                    loglik = np.sum(np.log(probs.sum(0)))
                    elbo[r,n] = loglik
                
            #    nparam = use_svgmm*nbin*4 + K*(X.shape[1] + X.shape[1]**2)/2
                maxlik = np.max([maxlik, loglik])
                if loglik == maxlik: # hold on to the best model
                    Mu, Sig, _, W = self.params[dset]
                    Pi = (np.squeeze(W.sum(0)/N).T)[:,np.newaxis]
                
                if self.verbose:
                    print('%.1f: Model %d/%d, run %d/%d' \
                          % (time.time() - t0, dset+1, num_dsets, r+1, n_runs))
                
            self.params[dset] = [Mu, Sig, Pi, W]
            
            if make_plots:
                dis = np.argmax(elbo[:,-1])
                plt.figure()
                plt.plot(elbo.T, c = [0.5,0.5,0.5],linewidth = 1)
                plt.plot(elbo[dis,:].T,'k-',linewidth = 2)
                plt.ylabel('log-likelihood')
                plt.title('Fits for dataset %d' % (dset + 1))
        
        if self.verbose:
            print('done')
        
    def E_Step(self, whichDset, dat):
        '''
        Does the E step in the EM algorithm. Updates the class assignments.
        '''
        
        means, covs, pi, _ = self.params[whichDset]
        
        log_w = np.array([sts.multivariate_normal.logpdf(dat,means[:,k],covs[:,:,k]) \
                          for k in range(self.num_comp[whichDset])])
        log_w += np.log(pi + 1e-16)
        c = log_w.max(0)
        log_norm = c + np.log(np.sum(np.exp(log_w - c), axis = 0))
        log_w -= log_norm
        w_ik = np.exp(log_w).T
        
        self.params[whichDset][-1] = w_ik
    
    def M_Step(self, whichDset, dat):
        '''
        Does the M step in the standard EM algorithm. 
        '''
        
        def my_cov(X, **kwargs):
            '''
            wrapper for covariance computation which rounds to nearest positive
            definite matrix
            '''
            sig_temp = np.cov(X, **kwargs)
            sig_r = r_mat_package.nearPD(sig_temp, eig_tol = 1e-6)
            sig_out = np.array(rob.r['as.matrix'](sig_r[0]))
            return sig_out
        
#        fudge = np.eye(self.dim_obs[whichDset])[:,:,np.newaxis]*1e-5
        
        # load class assignments and make them nice for broadcasting
        w_ik = self.params[whichDset][-1][:,np.newaxis,:]
        N = dat.shape[0]
        
        pi = np.tile(np.squeeze(w_ik.sum(0)/N),(N,1)).T
        
        mu = np.array([np.average(dat, axis = 0, weights = w_ik[:,0,k] + 1e-16) \
                       for k in range(self.num_comp[whichDset])]).T
        sig = np.array([my_cov(dat.T, aweights = w_ik[:,0,k] + 1e-16) \
                        for k in range(self.num_comp[whichDset])])
        sig = np.transpose(sig,(1,2,0))
        
        self.params[whichDset][0:-1] = [mu, sig, pi]
        
    def M_Step_SV(self, whichDset, dat, SV, nbin = 5, method = 'evenly'):
        '''
        Does the M step in the SVGMM EM algorithm. 
        '''
        
        def my_cov(X, **kwargs):
            '''
            wrapper for covariance computation which rounds to nearest positive
            definite matrix
            '''
            sig_temp = np.cov(X, **kwargs)
            sig_r = r_mat_package.nearPD(sig_temp, eig_tol = 1e-6)
            sig_out = np.array(rob.r['as.matrix'](sig_r[0]))
            return sig_out
        
        # load class assignments
        w_ik = self.params[whichDset][-1]
        
        pi = np.array(
                [self.neighbour_func(SV, w_ik[:,k], nbin = nbin, how = method)[0] \
                 for k in range(self.num_comp[whichDset])])
            
        mu = np.array([np.average(dat, axis = 0, weights = w_ik[:,k] + 1e-16) \
                       for k in range(self.num_comp[whichDset])]).T
        sig = np.array([my_cov(dat.T, aweights = w_ik[:,k] + 1e-16) \
                        for k in range(self.num_comp[whichDset])])
        sig = np.transpose(sig,(1,2,0))
        
        self.params[whichDset][0:-1] = [mu, sig, pi]
        
    def initialise_model(self, whichDset, dat, priors = None, 
                         how = 'randkm', s = 0.2):
        '''
        initialise_model(data, priors = None, how = 'randkm', s = 0.2)
        
        Initialise the model for one dataset.
        
        Args:
            whichDset (int): index of the dataset being initialised
            data ((n_obs,dim_obs) nparray): the data
            priors (list): [means, covariances, proportions]
            how ('prior' or 'randkm'): what method to use
            s (0-1): how much to perturb
        '''
        
        def my_cov(X, **kwargs):
            '''
            wrapper for covariance computation which rounds to nearest positive
            definite matrix
            '''
            sig_temp = np.cov(X, **kwargs)
            sig_r = r_mat_package.nearPD(sig_temp, eig_tol = 1e-6)
            sig_out = np.array(rob.r['as.matrix'](sig_r[0]))
            return sig_out
        
        if how is 'prior' and priors is None:
            raise ValueError('Must provide initial conditions for `prior` initialisation')
        
        n_obs, n_dim = dat.shape
        K = self.num_comp[whichDset]
        
        # Initialise parameters
        if n_dim != self.dim_obs[whichDset]:
            raise ValueError('Dimension of dataset doesn\'t match dim_obs')
        
        if how == 'prior': # perturbation from prior
            mu, sig, pi = priors
            
            mu_init = mu*(1 - s + 2*s*rnd.rand(1,K))
            sig_init = sig*(1 - s + 2*s*rnd.rand(1,1,K)) 
            pi_init = pi*(1 - s + 2*s*rnd.rand(K, 1))
            pi_init /= pi_init.sum(0)[:,np.newaxis].T
            
            distX = np.linalg.norm(dat[:,:,np.newaxis] \
                                   - mu_init[np.newaxis,:,:],axis = 1)
            z_init = np.zeros((n_obs,K))
            z_init[np.arange(n_obs), np.argmin(distX, axis = 1)] = 1 
            
        elif how == 'randkm': # sort of k-means thing
    #        if n_dim >= K:
            d = rnd.permutation(np.arange(n_dim))
    #        else:
    #            d = np.append(rnd.permutation(np.arange(n_dim)),
    #                          rnd.choice(n_dim, K-n_dim))
            whichdims = [d[0+n::K]\
                         for n in np.arange(K)]
    
            q = np.quantile(dat, [k*(1./K) \
                                  for k in range(K+1)], axis = 0)
            mean_qk = np.zeros((n_dim, K))
            
            for k in range(1,K+1): # get quantile means
                in_qk = [(dat[:,d] < q[k,d]) & (dat[:,d] >= q[k-1,d]) \
                         for d in range(n_dim)]
                mean_qk[:,k-1] = [dat[in_qk[d],d].mean(0) \
                        for d in range(n_dim)]
                
            mu_init = np.zeros((n_dim, K))
            for k in range(K): # select class means
                trailing = [whichdims[dd] \
                            for dd in rnd.permutation(np.setdiff1d(range(K),k))]
                nt = len(trailing)
                mu_init[whichdims[k],k] = mean_qk[whichdims[k],-1]
                for kk in range(nt):
                    mu_init[trailing[kk],k] = mean_qk[trailing[kk],-(kk+2)]
            mu_init *= (1 - s + 2*s*rnd.rand(1,K))
            
            distX = np.linalg.norm(dat[:,:,np.newaxis] - mu_init[np.newaxis,:,:],axis = 1)
            z_init = np.zeros((n_obs,K))
            z_init[np.arange(n_obs),np.argmin(distX, axis = 1)] = 1
            
            pi_init = np.tile(z_init.sum(0)/n_obs, (n_obs,1)).T
    
            sig_init = np.array([my_cov(dat.T, aweights = z_init[:,k] + 1e-16) \
                                 for k in range(K)]).T
        
        pi_init = pi_init.astype(np.float32)
        mu_init = mu_init.astype(np.float32)
        sig_init = sig_init.astype(np.float32)
        
        self.params[whichDset] = [mu_init, sig_init, pi_init, z_init]
        
    def neighbour_func(self,S, w_i, nb_func = np.mean, nbin = 5, how = 'evenly'):
        '''
        compute a function of w_i for each point in S within a binned neighbourhood
        
        wrapper for 'scipy.stats.binned_statistic_dd' function
        
        Args:
            S (N, dim_s): the spatial variable
            w_i (N,): the soft cluster assignment of each neuron 
            nb_func (callable; 1d --> scalars), default is np.mean: the function
                to compute
            nbin (int), default 5: number of spatial bins in each dimension
            how ({'evenly', 'quantiles'}), default 'evenly': make the bin edges
                evenly spaced or spaced as quantiles
        '''
        
        dim_s = S.shape[1]
        if how == 'evenly':
            bins = tuple([np.linspace(S[:,d].min(),S[:,d].max(),nbin+1) \
                          for d in range(dim_s)])
        elif how == 'quantiles':
            bins = tuple([np.quantile(S[:,d],np.arange(nbin+1)/nbin) \
                          for d in range(dim_s)])
            
        stat ,_, which_nb = sts.binned_statistic_dd(
                S, w_i, statistic = nb_func, bins = bins)
        
        if dim_s != 1:
            which_nb -= (nbin+3) # need to correct for silly indexing
            for b in range(nbin):
                deez = np.isin(which_nb,[range(b*(nbin+2),(b+1)*(nbin+2))])
                which_nb[deez] -= b*2
        else:
            which_nb -= 1
        
        pi_i = stat.flatten()[which_nb]
        
        return pi_i, which_nb
        
    def remove_outliers(self, X, k = 20, q = 0.1):
        
        nneigh = knn(k + 1)
        nneigh.fit(X)
        dist = nneigh.kneighbors(X,return_distance = True)[0][:,1:]
        dens = 1/np.mean(dist,axis = 1)
        
        keepers = (dens >= np.quantile(dens, q))
        
        return keepers
    
    #%% plotting
    def scatterplot(self, whichDset, dat, pdf_plot = 'contour',
                    c = None, cmap_scat = 'hsv', cmap_pdf = 'cool', 
                    alpha = 0.5, conf = 1./3.):
        '''
        Plot a dataset in a scatterplot matrix, along with the fit.
        '''
        
        self.E_Step(whichDset, dat)
        this_mu, this_sig, this_pi, this_w = self.params[whichDset]
        
        if c is not None:
            col = c
        else:
            col = self.soft_colormap(this_w, cmap_scat)
        
        df = pd.DataFrame(dat, columns = ('Trode1','Trode2','Trode3','Trode4'))
        axs = pdplt.scatter_matrix(df, s = 5, c = col, alpha = alpha, zorder = 2)
        plt.set_cmap(cmap_scat)
        
        for ii in range(self.dim_obs[whichDset]):
            for jj in range(self.dim_obs[whichDset]):
                if ii == jj:
                    continue
                
                if pdf_plot is 'density':
                    lims = np.array([axs[jj,ii].get_xlim(),axs[jj,ii].get_ylim()])    
                    C = this_sig[[ii,jj],:,:][:,[ii,jj]]
                    m = this_mu[[ii,jj],:]
                    pdf, Q = self.plot_2d_density(lims, m, C, this_pi)
                    axs[jj,ii].pcolormesh(Q[:,:,0],Q[:,:,1], pdf, 
                       cmap = cmap_pdf, zorder = 1)
                    
                elif pdf_plot is 'contour':
                    for k in range(self.num_comp[whichDset]):
                        C = this_sig[[ii,jj],:,k][:,[ii,jj]]
                        m = this_mu[[ii,jj],k]
                        ellipse_x, ellipse_y = self.cov_ellipse(C,m, conf = conf)
                        axs[jj,ii].plot(ellipse_x,ellipse_y, 'k-', linewidth = 1)
                    axs[jj,ii].scatter(this_mu[ii,:],this_mu[jj,:], 
                       c = 'k', marker = 'd')
                    
        return axs
                    
    def plot_2d_density(self,lims, mus, covs, pies, num = 100):
        '''
        mus (n_dim,K)
        covs (n_dim,n_dim,K)
        pies (1,K)
        '''
        
        K = mus.shape[1]
        
        Q = np.array(np.meshgrid(
                np.linspace(lims[0,0],lims[0,1],num),
                np.linspace(lims[1,0],lims[1,1],num))).T
        
        probs = np.array([pies[k]*sts.multivariate_normal.pdf(Q,mus[:,k],covs[:,:,k]) \
                          for k in range(K)])
        probs = probs.sum(0)
        
        return (probs, Q)
    
    def cov_ellipse(self,C, m, conf = 0.95):
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
    
    def soft_colormap(self, class_probs, cmap_name = 'jet', nbin = 200):
        '''
        Make a colormap which reflects soft cluster assignments. The hue says 
        which cluster a point is in, and saturation is the 'confidence' (0 when
        the maximum class_prob is 1/K, 1 when maximum class_prob is 1).
        '''
        N, K = class_probs.shape
        
        vals = np.argmax(class_probs, axis = 1)
        foo = cm.ScalarMappable(cmap = cmap_name)
        hsv = clr.rgb_to_hsv(foo.to_rgba(vals*(255/K))[:,:3])
    
        hsv[:,1] = (class_probs[range(N),vals] - (1/K))/(1 - (1/K))
        cols = clr.hsv_to_rgb(hsv)
        
        return cols

    #%% using
    def pdf(self, whichDset, values, use_log = False):
        '''
        Compute PDF of the specified mixture models. 
        
        Args:
            whichDset (int): which models to use
            values (nparray or list of nparray): values to evaluate models on
        '''
        
        if type(whichDset) is int:
            whichDset = [whichDset]
        elif type(values) is list and len(whichDset) is not len(values):
            raise ValueError('Length of "values" is not the same as "whichDset"')
        
        # we can evaluate the same dataset under multiple models
        if type(values) is not list:
            values = [values for _ in range(len(whichDset))]
        
        probs = []
        for dset, vals in zip(whichDset, values):
            
            this_mu, this_sig, this_pi, this_w = self.params[dset]
            
            if use_log:
                p = np.array([this_pi[k]*sts.multivariate_normal.logpdf(
                        vals, this_mu[:,k], this_sig[:,:,k]) \
                        for k in range(self.num_comp[dset])])
            else:
                p = np.array([this_pi[k]*sts.multivariate_normal.pdf(
                        vals, this_mu[:,k], this_sig[:,:,k]) \
                        for k in range(self.num_comp[dset])])
                    
            probs.append(p)
        
        return probs
    
    def logpdf(self, whichDset, values):
        '''
        Wrapper for pdf
        '''
        
        return self.pdf(whichDset, values, use_log = True)
    
    def compute_mark_probs(self, whichDset, marks, indices, final_shape):
        '''
        Makes the giant array with the probability of 
        
        whichDset (list of int): which models to compute with
        marks (list): mark values for each multiunit channel (nparray) or
            empty list for single unit channels
        indices (list): bin edges used to bin the spikes
        final_shape (tuple): shape of the desired array, usually is 
            (num_trials, num_time_pts, -1, dim_obs)
        
        TODO: be less terrible
        '''
        
        N = len(whichDset)
        num_t = final_shape[0]*final_shape[1]
        
        w = [[[] for _ in range(N)] for _ in range(num_t)]
        for n in whichDset:
            if marks[n] is not None:
                M = marks[n]
                inds = indices[n]
                pi_tet = self.params[n][2].T
                
                probs = self.logpdf([n],M)[0].T
            
            for t in range(num_t):
                if marks[n] is None:
                    w[t][n] = np.array([1])
                else:
                    Nt = np.sum(inds == t)
                    these_spks = probs[inds == t, :]
                    if Nt > 0: # renormalise for each spike
                        these_spks += np.log(pi_tet + 1e-16)
                        w_ik = np.sum(these_spks.T, axis = 1)
                        c = w_ik.max()
                        log_norm = c + np.log(np.sum(np.exp(w_ik - c)))
                        w_ik -= log_norm
                        w_ik = np.exp(w_ik)
                        w_ik[w_ik < 1e-12] = 0
                        w[t][n] = w_ik
                    else:
                        w[t][n] = np.squeeze(pi_tet)
        
        weight_array = np.array([self.build_weight_matrix(w[tt]) \
                                 for tt in range(num_t)])
        weight_array = np.reshape(weight_array,final_shape)
        
        return weight_array
    
    def build_weight_matrix(self,probs):
        '''
        probs (list) 
        '''
        n = len(probs)
        weight_mat = np.zeros((0,n))
        for tet in range(n):
            tmp = np.zeros((probs[tet].shape[0], n))
            tmp[:,tet] = probs[tet]
            weight_mat = np.append(weight_mat,tmp, axis = 0)
            
        return weight_mat
    
    #%% simulating
#    def generate_mixtures(self, whichDset):
#        '''
#        
#        '''
#        
#        if type(whichDset) is int:
#            whichDset = [whichDset]
#        
#        for dset in whichDset:
#        params = []
#        for tet in range(N_tet):
#            pi = rnd.dirichlet(np.ones(K[tet])*20)
#            mu= 200*np.array([rnd.rand(K[tet]) for _ in range(dim)]) + 50
#            
#            sig = np.zeros((dim,dim,K[tet]))
#            for k in range(K[tet]):
#                sig[:,:,k] = 20*sts.wishart.rvs(7,np.eye(dim))
#            params.append([mu, sig, pi])
#            
#        return params
    