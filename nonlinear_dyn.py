'''
Functions for simulating nonlinear dynamics
'''

def simulate_dpend(
        t0, tn, dt = 0.05, th1 = -180.0, w1 = 0.0, th2 = -130.0, w2 = 0.0, 
        G = 9.8, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0):
    '''
    th1 and th2 are the initial angles (degrees)
    w10 and w20 are the initial angular velocities (degrees per second)
    '''
    
    from numpy import sin, cos
    import numpy as np
    import scipy.integrate as integrate
    
    def derivs(state, t):
    
        dydx = np.zeros_like(state)
        dydx[0] = state[1]
    
        del_ = state[2] - state[0]
        den1 = (M1 + M2)*L1 - M2*L1*cos(del_)*cos(del_)
        dydx[1] = (M2*L1*state[1]*state[1]*sin(del_)*cos(del_) +
                   M2*G*sin(state[2])*cos(del_) +
                   M2*L2*state[3]*state[3]*sin(del_) -
                   (M1 + M2)*G*sin(state[0]))/den1
    
        dydx[2] = state[3]
    
        den2 = (L2/L1)*den1
        dydx[3] = (-M2*L2*state[3]*state[3]*sin(del_)*cos(del_) +
                   (M1 + M2)*G*sin(state[0])*cos(del_) -
                   (M1 + M2)*L1*state[1]*state[1]*sin(del_) -
                   (M1 + M2)*G*sin(state[2]))/den2
    
        return dydx
    
    # create a time array from 0..100 sampled at 0.05 second steps
    t = np.arange(t0, tn, dt)
    
    state = np.radians([th1, w1, th2, w2])

    # integrate your ODE using scipy.integrate.
    y = integrate.odeint(derivs, state, t)
    
    x1 = L1*sin(y[:, 0])
    y1 = -L1*cos(y[:, 0])
    
    x2 = L2*sin(y[:, 2]) + x1
    y2 = -L2*cos(y[:, 2]) + y1
    
    return np.array([x1,y1,x2,y2]).T

#%%
def simulate_walk(tn, dim = 2, dt = 0.05, win = 10, how = 'bounded', seed = None):
    '''
    simulate_walk(tn, dim = 2, dt = 0.05, win = 10)
    
    returns pos
    '''
    import numpy as np
    import numpy.random as rnd
    import scipy.special as spc
    
    rnd.seed(seed)
    # simulate position as random walk
    if how == 'unbounded': # simple and fast
        steps = dt*(rnd.rand(tn + win,dim) - 0.5)
        pos = np.cumsum(steps,axis = 0)
        for dd in range(dim):
            pos[:,dd] = np.convolve(pos[:,dd], np.ones(win), mode = 'same')
        pos = pos[:-win,:]
    else:
        tn += win
#        steps = dt*(rnd.rand(5*tn,dim) - 0.5)
        pos = np.zeros((tn,dim))
        for t in range(1,tn):
            gamma = rnd.gamma(np.sum(np.abs(pos[t-1,:])), 
                               scale = 0.5)
#            gamma = np.max([0,1-np.sum(np.abs(pos[t-1,:]))])
            W = dt*(rnd.rand(2,dim) - 0.5).sum(0)
            dp = W - dt*gamma*(pos[t-1,:])
            pos[t,:] = pos[t-1,:] + dp
            
        for dd in range(dim):
            pos[:,dd] = np.convolve(pos[:,dd], np.ones(win), mode = 'same')
        pos = pos[:-win,:]
    
    return pos

def simulate_responses(N_neur, pos, seed = None):
    '''
    simulate_responses(N_neur, pos, seed = None)
    
    returns (X, [cntrs, sigs])
    '''
    import numpy as np
    import numpy.random as rnd
    import scipy.stats as sts
    
    rnd.seed(seed)
    
    tn, dim = pos.shape
    
    lims = np.array([[pos[:,dd].min(), pos[:,dd].max()]\
                      for dd in range(dim)])
    
    # draw firing fields
#    mu_0 = lims.mean(1) + 0.2*(pos[0,:] - lims.mean(1))
#    sig_0 = 2*np.diag(np.abs(mu_0-pos[0,:]))
#    cntrs = rnd.multivariate_normal(mu_0, sig_0, size = N_neur)
    cntrs = rnd.uniform(lims[:,0].max(), lims[:,1].min(), size = (N_neur,2))
    
    sigs = np.zeros((dim,dim,N_neur))
    for n in range(N_neur):
#        sigs[:,:,n] = sts.wishart.rvs(4,0.5*np.eye(dim))/5
        sigs[:,:,n] = np.eye(dim)/3
    
    # evaluate rates for each neuron
#    mean_rate = rnd.gamma(1,4, N_neur)
    mean_rate = np.ones(N_neur)*10
    baseline = 0.01*rnd.gamma(1,1, N_neur)
    R = np.zeros((N_neur, tn))
    for n in range(N_neur):
        R[n,:] = mean_rate[n]*sts.multivariate_normal.pdf(pos,
         mean = cntrs[n,:], cov = sigs[:,:,n])
    
    X = rnd.poisson(R.T + baseline).T
    
    return (X, [cntrs, sigs])

