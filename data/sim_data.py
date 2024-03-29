"""Simulate data using the Model class in models.py"""

import numpy as np
import tensorflow as tf
from netlds.models import *
from netlds.generative import *
from netlds.inference import *

DTYPE = np.float32


def build_model(
        num_time_pts, dim_obs, dim_latent, dim_lps=None, num_layers=0,
        np_seed=0, tf_seed=0, obs_noise='gaussian',dynamics = 'linear',
        mark_probs = None):
    """
    Build netlds model to simulate data

    Args:
        num_time_pts (int): number of time points per trial
        dim_obs (int): number of observation dimensions
        dim_latent (int): number of latent space dimensions
        dim_lps (list of ints): dimension of each (optional) linear predictor
        num_layers (int): number of nn layers between latent space and
            observations
        np_seed (int, optional): numpy rng seed
        tf_seed (int, optional): tensorflow rng seed
        obs_noise (str, optional): distribution of observation noise
            'gaussian' | 'poisson'

    Returns:
        Model object

    """

    # for reproducibility
    np.random.seed(np_seed)
    tf.set_random_seed(tf_seed)

    # define model parameters
    if dynamics == 'linear':
        if dim_latent == 2:
            A = np.array([[ 0.9, -0.2], [ 0.2,  0.9]], dtype=DTYPE)
            z0_mean = np.array([[2.4, 2.3]], dtype=DTYPE)
        else:
            A = get_random_rotation_matrix(dim_latent)
            z0_mean = np.random.rand(1, dim_latent).astype(DTYPE)
    elif dynamics == 'identity':
        A = np.eye(dim_latent,dtype=DTYPE)
        z0_mean = np.random.rand(1, dim_latent).astype(DTYPE)
    elif dynamics == 'nonlinear':
        A = np.eye(dim_latent,dtype=DTYPE)
        z0_mean = np.random.rand(1, dim_latent).astype(DTYPE)

    Q = 0.03 * np.random.randn(dim_latent, dim_latent).astype(DTYPE)
    Q = np.matmul(Q, Q.T)
    Q_sqrt = np.linalg.cholesky(Q)
#
    if num_layers == 0:
        C = np.random.randn(dim_latent, dim_obs).astype(DTYPE)
        d = np.abs(np.random.randn(1, dim_obs).astype(DTYPE))
        gen_params = {
            'A': A, 'z0_mean': z0_mean, 'Q_sqrt': Q_sqrt, 'Q0_sqrt': Q_sqrt,
            'C': C, 'd': d}
        nn_params = None
    else:
        gen_params = {
            'A': A, 'z0_mean': z0_mean, 'Q_sqrt': Q_sqrt, 'Q0_sqrt': Q_sqrt}
        nn_params = []
        for i in range(num_layers):
            nn_params.append({
                'units': 15,
                'activation': 'tanh',
                'kernel_initializer': 'normal',
                'kernel_regularizer': None,
                'bias_initializer': 'zeros',
                'bias_regularizer': None})

    # specify inference network for approximate posterior
    inf_network = SmoothingLDS
    inf_network_params = {
        'dim_input': dim_obs,
        'dim_latent': dim_latent,
        'num_time_pts': num_time_pts}

    # specify probabilistic model
    if num_layers == 0:
        gen_model = LDS
    else:
        gen_model = FLDS

    if obs_noise is 'gaussian':
        R_sqrt = np.sqrt(0.05 * np.random.uniform(
            size=(1, dim_obs)).astype(DTYPE))
        gen_params['R_sqrt'] = R_sqrt

    gen_model_params = {
        'dim_obs': dim_obs,
        'dim_latent': dim_latent,
        'dim_predictors': dim_lps,
        'num_time_pts': num_time_pts,
        'gen_params': gen_params,
        'noise_dist': obs_noise,
        'nn_params': nn_params,
        'mark_probs': mark_probs}

    # initialize model
    model = LDSModel(
        inf_network=inf_network, inf_network_params=inf_network_params,
        gen_model=gen_model, gen_model_params=gen_model_params,
        couple_params=True)

    return model


def build_model_multi(
        num_time_pts, dim_obs, dim_latent, np_seed=0, tf_seed=0,
        obs_noise='gaussian'):
    """
    Build netlds model to simulate data

    Args:
        num_time_pts (int): number of time points per trial
        dim_obs (list of ints): number of observation dimensions for each pop
        dim_latent (list of ints): number of latent space dimensions for each
            pop
        np_seed (int, optional): numpy rng seed
        tf_seed (int, optional): tensorflow rng seed
        obs_noise (str, optional): distribution of observation noise
            'gaussian' | 'poisson'

    Returns:
        Model object

    """

    # for reproducibility
    np.random.seed(np_seed)
    tf.set_random_seed(tf_seed)

    dim_latent_all = sum(dim_latent)
    dim_obs_all = sum(dim_obs)
        
    # define model parameters
    A = get_random_rotation_matrix(dim_latent_all)
    z0_mean = np.random.rand(1, dim_latent_all).astype(DTYPE)

    Q = 0.01 * np.random.randn(dim_latent_all, dim_latent_all).astype(DTYPE)
    Q = np.matmul(Q, Q.T)
    Q_sqrt = np.linalg.cholesky(Q)

    C = []
    d = []
    for pop, _ in enumerate(dim_latent):
        C.append(np.random.randn(dim_latent[pop], dim_obs[pop]).astype(DTYPE))
        d.append(np.abs(np.random.randn(1, dim_obs[pop]).astype(DTYPE)))

    gen_params = {
        'A': A, 'z0_mean': z0_mean, 'Q_sqrt': Q_sqrt, 'Q0_sqrt': Q_sqrt,
        'C': C, 'd': d}

    # specify inference network for approximate posterior
    inf_network = SmoothingLDS
    inf_network_params = {
        'dim_input': dim_obs_all,
        'dim_latent': dim_latent_all,
        'num_time_pts': num_time_pts}

    # specify probabilistic model
    gen_model = NetLDS
    if obs_noise is 'gaussian':
        R_sqrt = []
        for _, pop_dim in enumerate(dim_obs):
            R_sqrt.append(np.sqrt(0.05 * np.random.uniform(
                size=(1, pop_dim)).astype(DTYPE)))
        gen_params['R_sqrt'] = R_sqrt

    gen_model_params = {
        'dim_obs': dim_obs,
        'dim_latent': dim_latent,
        'num_time_pts': num_time_pts,
        'noise_dist': obs_noise,
        'gen_params': gen_params}

    # initialize model
    model = LDSModel(
        inf_network=inf_network, inf_network_params=inf_network_params,
        gen_model=gen_model, gen_model_params=gen_model_params,
        couple_params=True)

    # actually initialize layers
    for pop, _ in enumerate(dim_latent):
        model.gen_net.networks[pop].params[0]['kernel_initializer'] = \
            tf.constant_initializer(gen_params['C'][pop])
        model.gen_net.networks[pop].params[0]['bias_initializer'] = \
            tf.constant_initializer(gen_params['d'][pop])

    return model


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

    return 0.95 * A.astype(DTYPE)
