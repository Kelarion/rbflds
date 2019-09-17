CODE_DIR = r'C:/Users/Matteo/Documents/GitHub/netlds/'
RESULTS_DIR_BASE = r'C:/Users/Matteo/Documents/UNI/Columbia/Rotation_Liam/'

import numpy as np
import numpy.random as rnd
import scipy
import scipy.special as spc
import scipy.stats as sts
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import sys
sys.path.insert(0, CODE_DIR)
from netlds.models import *
from netlds.generative import *
from netlds.inference import *
from data.sim_data import build_model
from nonlinear_dyn import simulate_dpend, simulate_walk, simulate_responses
from my_funcs import procrustes

#%%
def get_checkpoint_dir(network_, coupling_, batch_, mc_, train_A_):
    if network_.__name__ is 'SmoothingLDS':
        if train_A_:
            inf_net_str = 'lds-coupled'
        else:
            inf_net_str = 'lds-nodyn'
    elif network_.__name__ is 'MeanFieldGaussian':
        inf_net_str = 'mfg'
    elif network_.__name__ is 'MeanFieldGaussianTemporal':
        inf_net_str = 'mfgt'
    checkpoint_dir = str('%s_batch-%02i_samples-%02i/' % (inf_net_str, batch_, mc_))
    return checkpoint_dir

def get_random_rotation_matrix(dim):

    angle_in_deg = 40
    angle = angle_in_deg / 180.0 * np.pi
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])

    if dim == 2:
        A = rot
    else:
        # out = np.zeros((dim, dim))
        out = np.eye(dim)
        out[:2, :2] = rot
        q = np.linalg.qr(np.random.randn(dim, dim))[0]
        A = q.dot(out).dot(q.T)

    return 0.95 * A.astype(np.float32)
#%% functions for clusterless method
def generate_mixtures(K,N_tet,dim):
    '''
    K (list of ints) length 1 or N_tet
    '''
    if len(K) == 1:
        K = [K[0] for _ in range(N_tet)]
        
    params = []
    for tet in range(N_tet):
        pi = rnd.dirichlet(np.ones(K[tet])*20)
        mu= 200*np.array([rnd.rand(K[tet]) for _ in range(dim)]) + 50
        
        sig = np.zeros((dim,dim,K[tet]))
        for k in range(K[tet]):
            sig[:,:,k] = 20*sts.wishart.rvs(7,np.eye(dim))
        params.append([mu, sig, pi])
        
    return params

def build_w_matrix(probs):
    '''
    probs (list) 
    '''
    n = len(probs)
    W = np.zeros((0,n))
    for tet in range(n):
        tmp = np.zeros((probs[tet].shape[0], n))
        tmp[:,tet] = probs[tet]
        W = np.append(W,tmp, axis = 0)
        
    return W

def compute_mark_probs(mrks, gmm_params, final_shape):
    '''
    mrks (N-list) list of (marks, indices) for each tetrode
    gmm_params (N-list) parameters for gaussian mixture model
    
    TODO: be less terrible
    '''
    N = len(mrks)
    num_t = final_shape[0]*final_shape[1]
    
    w = [[[] for _ in range(N)] for _ in range(num_t)]
    for n in range(N):
        mu_tet = gmm_params[n][0]
        sig_tet = gmm_params[n][1]
        pi_tet = gmm_params[n][2]
        M = mrks[n][0]
        inds = mrks[n][1]
        K = len(pi_tet)
        
        probs = np.array([sts.multivariate_normal.logpdf(M,mu_tet[:,k],sig_tet[:,:,k]) \
                          for k in range(K)]).T
        
        for t in range(num_t):
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
                w[t][n] = pi_tet
    
    W = np.array([build_w_matrix(w[tt]) for tt in range(num_t)])
    W = np.reshape(W,final_shape)
    
    return W
    
#%%
# set simulation parameters
num_trials = 300
num_time_pts_ = [20,20,20]
dim_obs_ = [10,10,10]
dim_latent_ = [2,2,2]
dyn_type = ['nonlinear', 'nonlinear', 'nonlinear']
ys = [None, None, None]
zs = [None, None, None]
mrk = [None, None, None]
K_tet = [5] # number of components
dim_marks = 2
use_marks = [True, False, False]
is_sorted = [False, True, False]
sim_seed = 123

# set nonlinear dynamics beforehand
myz = []
for ii in range(len(dyn_type)):
    if dyn_type[ii] is 'nonlinear':
        rnd.seed(sim_seed)
        th1 = (2*rnd.randn(num_trials) - 180)
        rnd.seed(sim_seed)
        th2 = (2*rnd.randn(num_trials) - 130)
        my_dyn = np.zeros((num_trials, num_time_pts_[ii], dim_latent_[ii]))
        for jj in range(num_trials):
            my_dyn[jj,:,:] = simulate_dpend(0,num_time_pts_[0]*0.05,dt = 0.05,
                  th1 = th1[jj], th2 = th2[jj])[:,-2:]
    elif dyn_type[ii] is 'walk':
        my_dyn = simulate_walk(num_time_pts_[ii]*num_trials, seed = sim_seed)
        my_dyn = np.reshape(my_dyn, (num_trials, num_time_pts_[ii], dim_latent_[ii]))
    elif dyn_type[ii] is 'linear':
        my_dyn = None
    
    myz.append(my_dyn)

# make mark densities
gmm = generate_mixtures(K_tet, dim_obs_[0], dim_marks)
W = build_w_matrix([K_tet*gmm[ii][2] for ii in range(dim_obs_[0])])

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

#%%
mpr = np.tile(W, (num_trials,num_time_pts_[0],1,1))
obs_noise = 'poisson'
model = build_model(
        num_time_pts_[0], dim_obs_[0], dim_latent_[0], num_layers=2, np_seed=1,
        obs_noise=obs_noise, dynamics = dyn_type[0], mark_probs = mpr)

models = [[[[None for _ in range(num_batches)] \
                for _ in range(num_samples)] \
                for _ in range(num_inf_nets)] \
                for _ in range(len(dim_latent_))]

sim_num = 0
for num_time_pts, dim_obs, dim_latent in zip(num_time_pts_, dim_obs_, dim_latent_):
    sim_num += 1
    print('\n\nSIMULATION %i/%i' % (sim_num, len(dim_latent_)))
    
    if is_sorted[sim_num-1]:
        extra_str = 'sorted'
    else:
        if use_marks[sim_num-1]:
            extra_str = 'marked'
        else:
            extra_str = 'unmarked'
    
    RESULTS_DIR_SUFF = str('pflds_%02i-%02i-%02i-%s-%s/' 
                           % (num_time_pts, dim_obs, dim_latent, dyn_type[sim_num-1], extra_str))
    RESULTS_DIR = RESULTS_DIR_BASE + RESULTS_DIR_SUFF
    
    ##################
    # build simulation
    ##################
    if is_sorted[sim_num-1]:
        dim_obs = W.shape[0]
        mark_probs = None
        this_gmm = None
        myz[sim_num-1] = zs[sim_num-2]
    else:
        mark_probs = np.tile(W, (num_trials,num_time_pts,1,1))
        if use_marks[sim_num-1]:
            this_gmm = gmm
        else:
            this_gmm = None
    
    obs_noise = 'poisson'
#    model = build_model(
#        num_time_pts, dim_obs, dim_latent, num_layers=2, np_seed=1,
#        obs_noise=obs_noise, dynamics = dyn_type[sim_num-1], mark_probs = mark_probs)
    model.checkpoint_model(
        checkpoint_file=RESULTS_DIR + 'true_model.ckpt',
        save_filepath=True)
    ys[sim_num-1], zs[sim_num-1], mrk[sim_num - 1] = model.sample(num_samples=num_trials, seed=123,
      z_in = myz[sim_num-1], gmm_params = this_gmm, mark_probs = mark_probs)
    
    if use_marks[sim_num-1]:
        W_hat = compute_mark_probs(mrk[sim_num-1], 
                                   this_gmm, 
                                   (num_trials, num_time_pts, -1, dim_obs))
        num_clu = W_hat.shape[-2]
    else:
        W_hat = None
        num_clu = None
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
    noise_dist = obs_noise
    gen_model_params = {
        'dim_obs': dim_obs,
        'dim_latent': dim_latent,
        'num_time_pts': num_time_pts,
        'noise_dist': noise_dist,
        'nn_params': [{'units': 40}, {}], #  {'units': 15} # 3 layer nn to output
        'gen_params': None,
        'train_A': True,
        'train_Q0': True,
        'mark_probs': W_hat}
    
    # store models in nested list
#    models = [[[None for _ in range(num_batches)] \
#                for _ in range(num_samples)] \
#                for _ in range(num_inf_nets)]
    # initialize models
    for b in range(num_batches):
        for s in range(num_samples):
            for i in range(num_inf_nets):
                inf_network_params['num_mc_samples'] = mc_samples[s]
                gen_model_params['train_A'] = train_A[i]
#                gen_model_params['train_Q0'] = train_A[i]
                
                gen_model_params['gen_params'] = {
                        'A': np.eye(dim_latent, dtype=np.float32)}
#                gen_model_params['gen_params'] = {
#                        'A': np.eye(dim_latent, dtype=np.float32),
#                        'Q0_sqrt': 0.04*np.eye(dim_latent,dtype = np.float32)}
                
                models[sim_num-1][i][s][b] = LDSModel(
                    inf_network=inf_networks[i], inf_network_params=inf_network_params,
                    gen_model=gen_model, gen_model_params=gen_model_params,
                    np_seed=1, tf_seed=1, couple_params=coupling[i])
                
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
                    inf_networks[i], coupling[i], batch_sizes[b], mc_samples[s], train_A[i])
                opt_params['batch_size'] = batch_sizes[b]
                data_dict = {'observations': ys[sim_num-1],
                             'inf_input': ys[sim_num-1],
                             'mark_probs': W_hat}
                models[sim_num-1][i][s][b].train(
                    data=data_dict, opt_params=opt_params,
                    output_dir=RESULTS_DIR + checkpoint_dir)
                
                print('Done in %.2f seconds' % (time.time() - t0))
                
    print('\nFit all models in %.2f seconds' % (time.time() - t0))

#%% assess model performance
b = 0 # batch
s = 0 # sample
i = 0 # inference network
sim = 2 # simulation

num_time_pts = num_time_pts_[sim]
dim_latent = dim_latent_[sim]
dim_obs = dim_obs_[sim]

model_tr = models[sim][i][s][b]

input_data = ys[sim]

if is_sorted[sim]:
    extra_str = 'sorted'
else:
    if use_marks[sim]:
        extra_str = 'marked'
    else:
        extra_str = 'unmarked'
    
if use_marks[sim]:
    W_hat = compute_mark_probs(mrk[sim],
                               gmm, 
                               (num_trials, num_time_pts, -1, dim_obs))
else:
    W_hat = None

RESULTS_DIR_SUFF = str('pflds_%02i-%02i-%02i-%s-%s/' 
                       % (num_time_pts, dim_obs, dim_latent, dyn_type[sim], extra_str))
RESULTS_DIR = RESULTS_DIR_BASE + RESULTS_DIR_SUFF

#RESULTS_DIR_SUFF = str('pflds_%02i-%02i-%02i-%s-clusterless/' % (num_time_pts, dim_obs, dim_latent, dyn_type[sim]))
#RESULTS_DIR = RESULTS_DIR_BASE + RESULTS_DIR_SUFF
checkpoint_dir = get_checkpoint_dir(
    inf_networks[i], coupling[i], batch_sizes[b], mc_samples[s], train_A[i])
checkpoint_file = str('checkpoints/epoch_%05i.ckpt' % (opt_params['epochs_training'] - 1))
#
#model_tr.checkpoint_model(
#        checkpoint_file=RESULTS_DIR + checkpoint_dir + checkpoint_file,
#        save_filepath=True)
n= input_data.shape[-1]
y_mc, _, _ = model_tr.sample(ztype = 'posterior', y_in = input_data, mark_probs = W_hat)
yhat = np.squeeze(np.reshape(y_mc.mean(1),(1,-1,n)))
#yhat = y_mc.mean(1).mean(0)

ztmp = model_tr.get_posterior_means(
    input_data=input_data,
    checkpoint_file=RESULTS_DIR + checkpoint_dir + checkpoint_file,
    mark_probs = W_hat)

_, _, tform = procrustes(zs[sim].mean(0), ztmp.mean(0))
zhat = np.zeros(ztmp.shape)
for tt in range(num_trials):
    zhat[tt,:,:] = ztmp[tt,:,:].dot(tform['rotation'])*tform['scale'] + tform['translation']
    
fig = plt.figure()
gs = GridSpec(2,2)
ax0 = fig.add_subplot(gs[:,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,1])

ax0.set_title('Average reconstruction')
ax0.plot(np.squeeze(np.reshape(input_data,(1,-1,n))) + 10*np.arange(n))
ax0.set_prop_cycle(None)
ax0.plot(2*yhat + 10*np.arange(n), alpha = 0.5)
#ax0.plot(2*y1 + 10*np.arange(dim_obs), '--', alpha = 0.5 )

#ax0.plot(input_data.mean(0) + 10*np.arange(dim_obs))
#ax0.set_prop_cycle(None)
#ax0.plot(2*yhat + 10*np.arange(dim_obs), alpha = 0.5)

ax1.set_title('Inferred latent')
ax1.plot(zhat[:,:,0].T,zhat[:,:,1].T,c = '0.5',alpha = 0.1, zorder=1)
ax1.plot(zhat[:,:,0].mean(0),zhat[:,:,1].mean(0),c = 'k', zorder=2)
ax1.scatter(zhat[:,0,0],zhat[:,0,1], c = 'r', marker = 'd', zorder=3)

ax2.set_title('True latent')
ax2.plot(zs[sim][:,:,0].T,zs[sim][:,:,1].T, c = '0.5', alpha = 0.1, zorder=1)
ax2.plot(zs[sim][:,:,0].mean(0),zs[sim][:,:,1].mean(0), c = 'k', zorder=2)
ax2.scatter(zs[sim][:,0,0],zs[sim][:,0,1], c = 'r', marker = 'd', zorder=3)
#%% plot firing fields
n_grd = 50
cmap = 'bwr'

model_tr = models[sim][i][s][b]

# compute transformation between fit and true latent space

ztmp = model_tr.get_posterior_means(
        input_data=input_data,
        checkpoint_file=RESULTS_DIR + checkpoint_dir + checkpoint_file,
        mark_probs = W_hat)
_, z_plot, tform = procrustes(zs[sim].mean(0), ztmp.mean(0))
tform['rotation'] = np.linalg.inv(tform['rotation'])
tform['scale'] = 1/tform['scale']
tform['translation'] = -tform['translation']
#    tform = {'rotation': np.eye(2),'scale': 1.0, 'translation': 0}
#    z_plot = ztmp.mean(0)

# compute plotting parameters
nneur = model_tr.gen_net.networks[0].output_dim
nrow = int(np.sqrt(nneur))
ncol = int(nneur/nrow) + np.mod(nneur,nrow)

lims = [[1.2*z_plot[:,0].min(), 1.2*z_plot[:,0].max()],
        [1.2*z_plot[:,1].min(), 1.2*z_plot[:,1].max()]]

Q = np.array(np.meshgrid(np.linspace(lims[0][0],lims[0][1],n_grd),
                         np.linspace(lims[1][0],lims[1][1],n_grd))).T
vals = np.reshape(Q, (-1, 2))
vals = vals.dot(tform['rotation'])*tform['scale'] + tform['translation']
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
    ax.plot(z_plot[:,0],z_plot[:,1], 'k--')
    if r < nrow-1:
        ax.set_xticks([])
    if c > 0:
        ax.set_yticks([])

bigax.set_xlabel('Latent 1')
bigax.set_ylabel('Latent 2')
if true_model:
    bigax.set_title('True firing fields')
else:
    bigax.set_title('Fitted firing fields')

#%%
RESULTS_DIR_SUFF = str('pflds_%02i-%02i-%02i-%s/' % (num_time_pts, dim_obs, dim_latent, dyn_type[sim]))
RESULTS_DIR = RESULTS_DIR_BASE + RESULTS_DIR_SUFF
checkpoint_dir = get_checkpoint_dir(
    inf_networks[i], coupling[i], batch_sizes[b], mc_samples[s], train_A[i])
checkpoint_file = str('checkpoints/epoch_%05i.ckpt' % (opt_params['epochs_training'] - 1))

posterior_means = model_tr.get_posterior_means(
    input_data=input_data,
    checkpoint_file=RESULTS_DIR + checkpoint_dir + checkpoint_file)
avg_posterior_means = np.mean(posterior_means, axis=0)
z_avg = np.mean(zs[sim], axis=0)

import scipy as sp

plt.figure(figsize=(12, 4))

true_z, pred_z, _ = sp.spatial.procrustes(z_avg, avg_posterior_means)

num_rows = 2
num_cols = 3

# latent space
# for pop, pop_dim in enumerate(dim_latent):
#     # plot lvs individually
#     base_indx = sum(dim_latent[:pop])
#     for l in range(pop_dim):
#         plt.subplot(num_rows, num_cols, pop * num_cols + l + 1)
#         plt.plot(true_z[:, base_indx + l])
#         plt.plot(pred_z[:, base_indx + l])
#         plt.legend(('true', 'pred'))
#         plt.title('dim %1i' % (base_indx + l))

base_indx = 0
pop = 0
for l in range(dim_latent):
    plt.subplot(num_rows, num_cols, pop * num_cols + l + 1)
    plt.plot(true_z[:, base_indx + l])
    plt.plot(pred_z[:, base_indx + l])
    plt.legend(('true', 'pred'))
    plt.title('dim %1i' % (base_indx + l))
    
y_samps, z_samps = model_tr.sample(
    num_samples=128, 
    checkpoint_file=RESULTS_DIR + checkpoint_dir + checkpoint_file)

y_avg = np.mean(ys[sim], axis=0)
y_samps_avg = np.mean(y_samps, axis=0)

z_avg = np.mean(zs[sim], axis=0)
z_samps_avg = np.mean(z_samps, axis=0)

true_z_prior, pred_z_prior, _ = sp.spatial.procrustes(z_avg, z_samps_avg)

plt.figure(figsize=(12, 8))

num_rows = 4
num_cols = 3

# plot lvs individually
base_indx = 0
pop = 0
for l in range(dim_latent):
    plt.subplot(num_rows, num_cols, pop * num_cols * 2 + l + 1)
    plt.plot(pred_z_prior[:, base_indx + l])
    plt.plot(true_z_prior[:, base_indx + l]) #, color='black', linewidth=3)
    plt.legend(('pred', 'true'))
    plt.title('dim %1i' % (base_indx + l))
# plot some neurons individually
# num_neurons = 3
# for n in range(num_neurons):
#     plt.subplot(num_rows, num_cols, pop * num_cols * 2 + num_cols + 1 + n)
#     plt.plot(y_samps_avg[:, n])
#     plt.plot(y_avg[:, n]) #, color='black', linewidth=3)
#     plt.legend(('pred', 'true'))
#     plt.title('Pop %i - Neuron %i' % (pop, n))
        
plt.show()