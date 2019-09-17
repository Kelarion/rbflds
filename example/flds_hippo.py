CODE_DIR = r'C:/Users/Matteo/Documents/GitHub/netlds/'
RESULTS_DIR_BASE = r'C:/Users/Matteo/Documents/UNI/Columbia/Rotation_Liam/'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats as sts
from matplotlib.gridspec import GridSpec
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import time
import sys
import os
import glob
import pickle
sys.path.insert(0, CODE_DIR)
from netlds.models import *
from netlds.generative import *
from netlds.inference import *
from data.sim_data import build_model
from mixture_models import MixtureModel
from custom_layers import RBFLayer

numpy2ri.activate()
pc = importr('princurve',on_conflict = 'warn')

#%%
def get_checkpoint_dir(network_, coupling_, batch_, mc_, train_A_, rbf_):
    if network_.__name__ is 'SmoothingLDS':
        if train_A_:
            if rbf_:
                inf_net_str = 'lds-rbf-coupled'
            else:
                inf_net_str = 'lds-coupled'
        else:
            if rbf_:
                inf_net_str = 'lds-rbf-nodyn'
            else:
                inf_net_str = 'lds-nodyn'
    elif network_.__name__ is 'MeanFieldGaussian':
        inf_net_str = 'mfg'
    elif network_.__name__ is 'MeanFieldGaussianTemporal':
        inf_net_str = 'mfgt'
    checkpoint_dir = str('%s_batch-%02i_samples-%02i/' % (inf_net_str, batch_, mc_))
    return checkpoint_dir

def which_side(z,princ,inds):
    '''
    On which side of the projection curve `princ` do the points `z` lie? 
    returns +/- 1
    '''
    epsl = (z - princ)/np.sqrt(((z - princ)**2).sum(1))[:,np.newaxis]
    
    epsl_ord = epsl[inds-1,:]
    dots = (epsl_ord[1:,:]*epsl_ord[:-1,:]).sum(1)
    sgn_tmp = np.append(dots[0],dots)
    for ii in range(len(sgn_tmp) - 1):
        sgn_tmp[ii+1] = sgn_tmp[ii]*sgn_tmp[ii+1]
    sng = np.zeros(z.shape[0])
    sng[inds-1] = np.sign(sgn_tmp)
    
    return sng

def initialize_means_2d(n_neur, scales = [1,1]):
    
    n1 = int(n_neur/2)
    n2 = n_neur - n1
    
    grp1 = np.exp(1j*np.arange(n1)*(2*np.pi/n1))
    grp2 = np.exp(1j*np.arange(n2)*(2*np.pi/n2))
    
    x = np.concatenate([np.real(grp1), np.real(grp2)/np.pi])*scales[0]
    y = np.concatenate([np.imag(grp1), np.imag(grp2)/np.pi])*scales[1]
    
    return np.array([x,y])

#%%
# load data
data_dirs = os.listdir(RESULTS_DIR_BASE + r'/data')
binsize = 0.2 # 200 ms bins
epoch = [4564,5501]
whichEpoch = 'epoch4'
use_all = True

bins = np.arange(epoch[0],epoch[1],binsize)

#st = np.load(r'C:\Users\Matteo\Documents\UNI\Columbia\Rotation_Liam\data\sorted.spike.time2.npy')
#clu = np.load(r'C:\Users\Matteo\Documents\UNI\Columbia\Rotation_Liam\data\sorted.spike.cluster2.npy')
#cids = np.unique(clu)
#Y = np.array([np.histogram(st[clu == c],bins = bins)[0] for c in cids])

# for loading the raw data (?)
nbin = bins.shape[0] - 1
Y = np.empty((0,nbin))
whichClu = np.empty(0)
st = np.empty(0)
st_multi = np.empty(0)
clu = np.empty(0)
dset = np.empty(0)
whichBin = np.empty(0)
mua_dset = np.empty(0)
amps = np.empty((0,4))
ii = 0
for dd in data_dirs:
    ii += 1
    if ('.mat' in dd) or ('.npy' in dd) or ('.pkl' in dd):
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
    st_multi = np.append(st_multi, spks_tmp[good_mua]/10000)
    mua_dset = np.append(mua_dset, np.ones(np.sum(good_mua))*trode)
    whichBin = np.append(whichBin, 
                         np.digitize(spks_tmp[good_mua]/10000, bins))
    clus_tmp[good_mua] = -trode - m
    
    amps = np.append(amps,feat_tmp[good_mua,-4:],axis = 0)
    
    if use_all:
        use_spks = (clus_tmp[:,0] != 0) | good_mua
    else:  
        use_spks = clus_tmp > 0
    
    st = np.append(st,spks_tmp[use_spks]/10000)
    clu = np.append(clu,clus_tmp[use_spks] + m)
    dset = np.append(dset,np.ones(np.sum(use_spks))*trode)
    cids = np.unique(clus_tmp[use_spks])
    if cids.size > 0:
        sp = np.array([np.histogram(spks_tmp[clus_tmp == c]/10000,bins = bins)[0] for c in cids])
        Y = np.append(Y, sp, axis = 0)
        whichClu = np.append(whichClu,cids+m)
        print(dd + ': done')
        
clu = clu[np.argsort(st)]  
dset = dset[np.argsort(st)]
whichBin = whichBin[np.argsort(st_multi)]
mua_dset = mua_dset[np.argsort(st_multi)]
amps = amps[np.argsort(st_multi),:]
st = np.sort(st)
st_multi = np.sort(st_multi)

Y = Y[np.argsort(whichClu),:]
whichClu = np.sort(whichClu)

pos = np.load(r'C:\Users\Matteo\Documents\UNI\Columbia\Rotation_Liam\data\behave.position4.npy')
t_pos = np.load(r'C:\Users\Matteo\Documents\UNI\Columbia\Rotation_Liam\data\behave.time4.npy')
#%% Rearrange the data
nT = 20 # time points per 'trial'

whichClu = whichClu[Y.sum(1) > 100]
Y = Y[Y.sum(1) > 100,:]

Y = np.transpose(Y[:,:,np.newaxis],(2,1,0))

tau = np.mod(nbin,nT)
r = int((nbin - tau)/nT)
Y = np.reshape(Y[:,tau:,:],(r,nT,-1))

bins = bins[1+tau:]

print(Y.shape) # should be (r,nT,nneur)

num_trials = Y.shape[0]
num_time_pts = Y.shape[1]
dim_obs = Y.shape[2]

#%% Fit initially
# set fitting parameters
dim_latent = 2

# define model types to loop over
batch_sizes = [1]
mc_samples = [32]
inf_networks = [SmoothingLDS]
# couple inference network and model parameters, only valid for SmoothingLDS
coupling = [True]
train_A = [False]  
num_batches = len(batch_sizes)
num_samples = len(mc_samples)
num_inf_nets = len(inf_networks)

# set optimization parameters
adam = {'learning_rate': 1e-3}
opt_params = {
    'learning_alg': 'adagrad',
    'adam': adam,
    'epochs_training': 500,    # max iterations
    'epochs_display': None,    # output to notebook
    'epochs_ckpt': np.inf,     # checkpoint model parameters (inf=last epoch)
    'epochs_summary': 5,       # output to tensorboard
    'batch_size': None,        # changes 
    'use_gpu': False,           
    'run_diagnostics': False}  # memory/compute time in tensorboard
    
###################################
# initialize models with same seeds
###################################  
# specify inference network for approximate posterior
inf_network_params = {
    'dim_input': dim_obs,
    'dim_latent': dim_latent,
    'num_mc_samples': None,
    'num_time_pts': num_time_pts}

# specify probabilistic model 
gen_model = FLDS
noise_dist = 'poisson'
gen_model_params = {
    'dim_obs': dim_obs,
    'dim_latent': dim_latent,
    'num_time_pts': num_time_pts,
    'noise_dist': noise_dist,
    'nn_params': [{'units': 50}, {}],  # [{'units': 15}, {'units': 15}, {}] 3 layer nn to output
    'gen_params': None,
    'train_A': None,
    'train_Q0': None}

# Fit 
RESULTS_DIR_SUFF = str('pflds_firstpass_%02i-%02i-%02i/' 
                       % (num_time_pts, dim_obs, dim_latent))
RESULTS_DIR = RESULTS_DIR_BASE + RESULTS_DIR_SUFF

fp_model = []
    
# initialize models
inf_network_params['num_mc_samples'] = mc_samples[0]
gen_model_params['train_A'] = train_A[0]
gen_model_params['train_Q0'] = train_A[0]
    
if train_A[0]:
    gen_model_params['gen_params'] = None
else:
    gen_model_params['gen_params'] = {
            'A': np.eye(dim_latent,dtype = np.float32),
            'Q0_sqrt': 5*np.eye(dim_latent,dtype = np.float32)}

fp_model = LDSModel(
    inf_network=inf_networks[0], inf_network_params=inf_network_params,
    gen_model=gen_model, gen_model_params=gen_model_params,
    couple_params=coupling[0])

#######
# train
#######
t0 = time.time()
print('\n\nFitting first-pass model...')

# get checkpoint dir from model specs
checkpoint_dir = get_checkpoint_dir(
        SmoothingLDS, True, batch_sizes[0], mc_samples[0], train_A[0], False)
opt_params['batch_size'] = batch_sizes[0]
data_dict = {'observations': Y, 'inf_input': Y}
fp_model.train(
    data=data_dict, opt_params=opt_params,
    output_dir=RESULTS_DIR + checkpoint_dir)
print('Done in %.2f seconds' % (time.time() - t0))

checkpoint_file = str('checkpoints/epoch_%05i.ckpt' % 
                      (opt_params['epochs_training'] - 1))
zhat = fp_model.get_posterior_means(
    input_data=Y,
    checkpoint_file=RESULTS_DIR + checkpoint_dir + checkpoint_file)
z_firstpass = np.reshape(zhat,(1,-1,dim_latent)).squeeze()

S = np.zeros((st_multi.shape[0], dim_latent))
for dim in range(dim_latent):
    S[:,dim] = np.interp(st_multi, bins, z_firstpass[:,dim])

np.save(RESULTS_DIR_BASE + r'/data/' + 'unsorted.spike.latent.npy',S)
#%% Organise the marks
n_cluster = 5
S = np.load(RESULTS_DIR_BASE + r'/data/' + 'unsorted.spike.latent.npy')

in_epoch = (st_multi > epoch[0])&(st_multi < epoch[1])

marks = []
indices = []
spatial_var = []
dim_marks = []
K_tetrode = []
for iclu in whichClu:
    if iclu>0:
        marks.append(None)
        indices.append(None)
        spatial_var.append(None)
        dim_marks.append(None)
        K_tetrode.append(1)
    else:
        nspk = np.sum(in_epoch & (mua_dset == -iclu))
        marks.append(amps[in_epoch & (mua_dset == -iclu),:])
        indices.append(whichBin[in_epoch & (mua_dset == -iclu)])
        spatial_var.append(S[in_epoch & (mua_dset == -iclu),:])
        dim_marks.append(amps.shape[1])
        if nspk < 500:
            K_tetrode.append(2)
        elif nspk < 1000:
            K_tetrode.append(3)
        elif nspk < 10000:
            K_tetrode.append(4)
        else:
            K_tetrode.append(5)

single_units = list(np.where(whichClu > 0)[0])
multi_units = list(np.where(whichClu < 0)[0])

mark_mdl = MixtureModel(dim_marks, K_tetrode)

#%% Fit mark model and apply
print('Fitting mixture models, ')
mark_mdl.fit_model(marks, max_iter = 150, n_runs = 100, skip_datasets = single_units,
                   use_spatial = True, spatial_variable = spatial_var,
                   init_method = 'prior', num_spatial_bins = 3)

with open(RESULTS_DIR_BASE + r'/data/mark_model.pkl','wb') as fil:
    pickle.dump(mark_mdl,fil, pickle.HIGHEST_PROTOCOL)

#%%
with open(RESULTS_DIR_BASE + r'/data/mark_model.pkl','rb') as fil:
    mark_mdl = pickle.load(fil)
    
print('now computing mark probabilities')
W_hat = mark_mdl.compute_mark_probs(range(dim_obs), marks, indices,
                                    (num_trials, num_time_pts, -1, dim_obs))
num_clu = W_hat.shape[-2]
print('Done.')
#%% Fit for real
# set fitting parameters
dim_latent = 2
num_clu = None

# define model types to loop over
batch_sizes = [1]
mc_samples = [16]
inf_networks = [SmoothingLDS]
# couple inference network and model parameters, only valid for SmoothingLDS
coupling = [True]
train_A = [False]
use_RBF = [True]
num_batches = len(batch_sizes)
num_samples = len(mc_samples)
num_inf_nets = len(inf_networks)

# network initialisers
I = tf.initializers.constant(np.eye(dim_obs))

# set optimization parameters
adam = {'learning_rate': 1e-3}
opt_params = {
    'learning_alg': 'adagrad',
    'adam': adam,
    'epochs_training': 2000,    # max iterations
    'epochs_display': None,    # output to notebook
    'epochs_ckpt': np.inf,     # checkpoint model parameters (inf=last epoch)
    'epochs_summary': 5,       # output to tensorboard
    'batch_size': None,        # changes 
    'use_gpu': False,           
    'run_diagnostics': False}  # memory/compute time in tensorboard
    
###################################
# initialize models with same seeds
###################################  
# specify inference network for approximate posterior
inf_network_params = {
    'dim_input': dim_obs,
    'dim_latent': dim_latent,
    'num_mc_samples': None,
    'num_time_pts': num_time_pts,
    'num_clusters': num_clu}

# specify probabilistic model 
gen_model = FLDS
noise_dist = 'poisson'
gen_model_params = {
    'dim_obs': dim_obs,
    'dim_latent': dim_latent,
    'num_time_pts': num_time_pts,
    'noise_dist': noise_dist,
    'nn_params': [{'units': 50}, {}],  # [{'units': 15}, {'units': 15}, {}] 3 layer nn to output
    'gen_params': None,
    'train_A': None,
    'train_Q0': None,
    'num_clusters': num_clu}

# In[71]:
RESULTS_DIR_SUFF = str('pflds_%02i-%02i-%02i_2/' % (num_time_pts, dim_obs, dim_latent))
RESULTS_DIR = RESULTS_DIR_BASE + RESULTS_DIR_SUFF

# store models in nested list
models = [[[None for _ in range(num_batches)]
            for _ in range(num_samples)]
            for _ in range(num_inf_nets)]

# initialize models
for b in range(num_batches):
    for s in range(num_samples):
        for i in range(num_inf_nets):
            inf_network_params['num_mc_samples'] = mc_samples[s]
            gen_model_params['train_A'] = train_A[i]
            gen_model_params['train_Q0'] = train_A[i]
            
            if train_A[i]:
                gen_model_params['gen_params'] = None
            else:
                gen_model_params['gen_params'] = {
                        'A': np.eye(dim_latent,dtype = np.float32),
                        'Q0_sqrt': 10*np.eye(dim_latent,dtype = np.float32)}
                
            if not use_RBF[i]:
                gen_model_params['nn_params'] = [{'units': 50},{}]
            else:
                gen_model_params['nn_params']= \
                [{'layer_type': RBFLayer,
                  'sigma': 0.05,
                  'units': dim_obs,
                  'trainable': True},
                 {'kernel_initializer': I,
                   'activation': 'identity',
                   'trainable': False}]
#                inf_network_params['nn_params']= \
#                [{},
#                 {'layer_type': RBFLayer,
#                  'sigma': 0.1,
#                  'units': dim_obs,
#                  'trainable': True}, {}]
            models[i][s][b] = LDSModel(
                inf_network=inf_networks[i], inf_network_params=inf_network_params,
                gen_model=gen_model, gen_model_params=gen_model_params,
                couple_params=coupling[i])

#######
# train
#######
t0 = time.time()
model_counter = 0
total_models = num_batches * num_samples * num_inf_nets
for b in range(num_batches):
    for s in range(num_samples):
        for i in range(num_inf_nets):

            model_counter += 1
            print('\n\nFitting model %i/%i' % (model_counter, total_models))

            # get checkpoint dir from model specs
            checkpoint_dir = get_checkpoint_dir(
                inf_networks[i], coupling[i], batch_sizes[b], mc_samples[s], 
                train_A[i], use_RBF[i])
            opt_params['batch_size'] = batch_sizes[b]
#            data_dict = {'observations': Y, 'inf_input': Y, 'mark_probs': W_hat}
            data_dict = {'observations': Y, 'inf_input': Y}
            models[i][s][b].train(
                data=data_dict, opt_params=opt_params,
                output_dir=RESULTS_DIR + checkpoint_dir)
            models[i][s][b].save_model(RESULTS_DIR + 'model_%i_%i_%i' % (i,s,b))
            print('Done in %.2f seconds' % (time.time() - t0))

print('\nFit all models in %.2f seconds' % (time.time() - t0))

#%% get mean of posterior
b = 0
s = 0
i = 0

model_tr = models[i][s][b]

#RESULTS_DIR_SUFF = str('pflds_%02i-%02i-%02i_2/' % (num_time_pts, dim_obs, dim_latent))
RESULTS_DIR = RESULTS_DIR_BASE + RESULTS_DIR_SUFF
checkpoint_dir = get_checkpoint_dir(
    inf_networks[i], coupling[i], batch_sizes[b], mc_samples[s], train_A[i], use_RBF[i])
checkpoint_file = str('checkpoints/epoch_%05i.ckpt' % (opt_params['epochs_training'] - 1))

input_data = Y

#y_mc, _, _ = model_tr.sample(ztype = 'posterior', y_in = input_data,
#                             mark_probs = W_hat)
y_mc, _, _ = model_tr.sample(ztype = 'posterior', y_in = input_data)
#yhat = np.squeeze(np.reshape(y_mc.mean(1),(1,-1,dim_obs)))

#zhat = model_tr.get_posterior_means(
#    input_data=input_data,
#    checkpoint_file=RESULTS_DIR + checkpoint_dir + checkpoint_file,
#    mark_probs = W_hat)
zhat = model_tr.get_posterior_means(
    input_data=input_data,
    checkpoint_file=RESULTS_DIR + checkpoint_dir + checkpoint_file)
z_bar = np.reshape(zhat,(1,-1,dim_latent)).squeeze()

#%% plot with reconstruction
fig = plt.figure()
gs = GridSpec(1,2)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])

ax0.set_title('Mean reconstruction')
ax0.plot(np.squeeze(np.reshape(input_data,(1,-1,dim_obs))) + 10*np.arange(dim_obs))
ax0.set_prop_cycle(None)
ax0.plot(yhat + 10*np.arange(dim_obs), alpha = 0.5)

ax1.set_title('Inferred latent')
ax1.plot(zhat[:,:,0].T,zhat[:,:,1].T,c = '0.5',alpha = 0.5, zorder=1)
ax1.plot(zhat[:,:,0].mean(0),zhat[:,:,1].mean(0),c = 'k', zorder=2)
ax1.scatter(zhat[:,0,0],zhat[:,0,1], c = 'r', marker = 'd', zorder=3)

#%% plot with place

these_clu = np.array([(c in whichClu) for c in clu])
this_dset = 12

st_epoch = st[(st > epoch[0])&(st < epoch[1]) & these_clu]
clu_epoch = clu[(st > epoch[0])&(st < epoch[1]) & these_clu]
mua_epoch = st_multi[(st_multi > epoch[0])&(st_multi < epoch[1]) & (mua_dset == this_dset)]

z1_spk = np.interp(st_epoch, bins, z_bar[:,0])
z2_spk = np.interp(st_epoch, bins, z_bar[:,1])

z1_multi = np.interp(mua_epoch, bins, z_bar[:,0])
z2_multi = np.interp(mua_epoch, bins, z_bar[:,1])

x_spk = np.interp(st_epoch, t_pos[:,0], pos[:,0])
y_spk = np.interp(st_epoch, t_pos[:,0], pos[:,1])
x_multi = np.interp(mua_epoch, t_pos[:,0], pos[:,0])
y_multi = np.interp(mua_epoch, t_pos[:,0], pos[:,1])

z = np.array([z1_spk,z2_spk]).T
x = np.array([x_spk,y_spk]).T

nt = np.logical_not # why is this function so clunky??

arm1 = (x_spk < 198) & (y_spk > 80)
goal1 = arm1 & (y_spk > 120)

arm2 = nt(arm1) & (x_spk < 235) & (y_spk > 80)
goal2 = arm2 & (y_spk > 120)

arm3 = nt(arm1) & nt(arm2) & (y_spk > 80)
goal3 = arm3 & (y_spk > 120)

trunk = nt(arm1) & nt(arm2) & nt(arm3)

whichArm = np.zeros(x_spk.shape)
whichArm[arm1 & nt(goal1)] = 1
whichArm[goal1] = 1.5
whichArm[arm2 & nt(goal2)] = 2
whichArm[goal2] = 2.5
whichArm[arm3 & nt(goal3)] = 3
whichArm[goal3] = 3.5

#these_spk = (clu_epoch ==16)
these_spk = np.ones(st_epoch.shape) > 0

#col = lam[these_spk]
#col = dist[these_spk]
col = whichArm
#col = y_spk
cm = 'jet'
nt = np.logical_not

#%%
#fig, axs = plt.subplots(1,2, figsize = (15,7))
plt.figure(figsize = (15,15))
plt.tight_layout()
plt.set_cmap(cm)

#axs[0].scatter(x[nt(these_spk),0],x[nt(these_spk),1],s = 1, c = '0.7',alpha = 0.5)
#axs[0].scatter(x[these_spk,0],x[these_spk,1],s = 3, c = col)
#axs[1].scatter(z[nt(these_spk),0],z[nt(these_spk),1],s = 1, c = '0.7',alpha = 0.5)
#axs[1].scatter(z[these_spk,0],z[these_spk,1],s = 3, c = col)

plt.scatter(z[nt(these_spk),0],z[nt(these_spk),1],s = 1, c = '0.7',alpha = 0.5)
plt.scatter(z[these_spk,0],z[these_spk,1],s = 3, c = col)

#axs[0].set_xlabel('horizontal pos')
#axs[0].set_ylabel('vertical pos')
#axs[1].set_xlabel('latent 1')
#axs[1].set_ylabel('latent 2')
#axs[0].set_title('spike position')
#axs[1].set_title('spike latent')
#%%
n_grd = 50
cmap = 'bwr'

model_tr = models[i][s][b]

# compute transformation between fit and true latent space

#ztmp = model_tr.get_posterior_means(
#        input_data=input_data,
#        checkpoint_file=RESULTS_DIR + checkpoint_dir + checkpoint_file)
z_plot = z_bar

# compute plotting parameters
nneur = model_tr.gen_net.networks[0].output_dim
nrow = int(np.sqrt(nneur))
ncol = int(nneur/nrow) + np.mod(nneur,nrow)

lims = [[1.2*z_plot[:,0].min(), 1.2*z_plot[:,0].max()],
        [1.2*z_plot[:,1].min(), 1.2*z_plot[:,1].max()]]

Q = np.array(np.meshgrid(np.linspace(lims[0][0],lims[0][1],n_grd),
                         np.linspace(lims[1][0],lims[1][1],n_grd))).T
vals = np.reshape(Q, (-1, 2))
fld = model_tr.joint_intensity(vals)
fld /= fld.max(0)
fields = np.reshape(fld,(n_grd,n_grd,nneur))

# plot the fields
fig = plt.figure()
bigax = fig.add_subplot(111)
bigax.spines['top'].set_color('none')
bigax.spines['bottom'].set_color('none')
bigax.spines['left'].set_color('none')
bigax.spines['right'].set_color('none')
bigax.tick_params(labelcolor='w', top=False, 
                  bottom=False, left=False, right=False)
plt.set_cmap(cmap)
gs = GridSpec(nrow,ncol)
for nn in range(nneur):
    r = int(nn/ncol)
    c = np.mod(nn,ncol)
    
    ax = fig.add_subplot(gs[r,c])
    ax.pcolormesh(Q[:,:,0],Q[:,:,1],fields[:,:,nn])
    ax.plot(z_plot[:,0],z_plot[:,1], 'k--', alpha = 0.1)
    if r < nrow-1:
        ax.set_xticks([])
    if c > 0:
        ax.set_yticks([])

bigax.set_xlabel('Latent 1')
bigax.set_ylabel('Latent 2')
bigax.set_title('Fitted firing fields')


#%%

#%%
# get the projection line for discriminating goal and trunk
m1 = np.mean(z[np.logical_or(goal1,goal2,goal3),:],axis = 0)
m2 = np.mean(z[trunk,:],axis = 0)
LD = (m2 - m1)/np.linalg.norm(m2 - m1)
P = np.outer(LD,LD) # projector onto linear discriminant
p = P.dot(z.T).T
    
#%%
dest_dir = glob.glob(RESULTS_DIR_BASE + r'/data/' + str(this_dset) + '*')[0]

z_save = np.array([z1_multi,z2_multi]).T
#x_save = np.array([x_multi,y_multi]).T

#fitt = pc.principal_curve(z_save)
#princ = np.array(list(fitt)[0])
#inds = np.array(list(fitt)[1])
#lam = np.array(list(fitt)[2])
#dist = np.sqrt(((z_save - princ)**2).sum(1)) * which_side(z_save,princ,inds)

proj = P.dot(z_save.T).T
LD_val = np.linalg.norm(proj - proj[np.argmin(proj[:,0]),:],axis = 1)

fname_root = r'/unsorted.spike.%s' % whichEpoch
#np.save(dest_dir + fname_root + '.latentPCline.npy',princ[inds-1,:])
#np.save(dest_dir + fname_root + '.latentPC.npy',lam)
#np.save(dest_dir + fname_root + '.latent.npy',z)
np.save(dest_dir + fname_root + '.pos.npy', np.array([x_multi,y_multi]).T)
np.save(dest_dir + fname_root + '.LDA_value.npy', LD_val)

# In[80]:

fig, axs = plt.subplots(1,2,figsize = (9.6,4.31))
axs[0].set_xlim(np.min(pos[:,0]),np.max(pos[:,0]))
axs[0].set_ylim(np.min(pos[:,1]),np.max(pos[:,1]))
axs[1].set_xlim(np.min(z1_spk),np.max(z1_spk))
axs[1].set_ylim(np.min(z2_spk),np.max(z2_spk))

axs[0].grid()

siz = np.zeros(x_spk.shape)
posline, = axs[0].plot([], [], '--', lw=1, color = (0.9,0.9,0.9))
pscat = axs[0].scatter(x_spk, y_spk, s = siz, c = clu_epoch)
zscat = axs[1].scatter(z1_spk, z2_spk, s = siz, c = clu_epoch)
time_template = 'time = %.1fs'
time_text = axs[0].text(0.05, 0.9, '', transform=axs[0].transAxes)

def init():
    posline.set_data([], [])
    pscat.set_sizes(siz) 
    zscat.set_sizes(siz)
    time_text.set_text('')
    return posline, pscat, zscat, time_text

def animate(i):
    thisx = pos[:i,0]
    thisy = pos[:i,1]
    
    these_st = (st_epoch <= t_pos[i])

    posline.set_data(thisx, thisy) 
    
    siz[these_st] = 1
    pscat.set_sizes(siz)
    zscat.set_sizes(siz)
    
    time_text.set_text(time_template % (t_pos[i]))
    return posline, pscat, zscat, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(len(t_pos)),
                              interval=3.3, blit=True, init_func=init)

ani.save('spikes_pos_latent.mp4', fps=30)
