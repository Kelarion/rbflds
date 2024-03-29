"""Model class for building models"""

import os
import numpy as np
import numpy.random as rnd
import scipy.stats as sts
import tensorflow as tf
from netlds.generative import *
from netlds.inference import *
from netlds.trainer import Trainer


class Model(object):
    """Base class for models"""

    # use same data type throughout graph construction
    dtype = tf.float32

    def __init__(
            self, inf_network=None, inf_network_params=None, gen_model=None,
            gen_model_params=None, np_seed=0, tf_seed=0):
        """
        Constructor for full Model; combines an inference network with a
        generative model and provides training functions

        Args:
            inf_network (InferenceNetwork class)
            inf_network_params (dict)
            gen_model (GenerativeModel class)
            gen_model_params (dict)
            np_seed (int): for training minibatches
            tf_seed (int): for initializing tf.Variables (sampling functions
                have their own seed arguments)

        """

        # initialize inference network and generative models
        self.inf_net = inf_network(**inf_network_params)
        self.gen_net = gen_model(**gen_model_params)

        # initialize Trainer object
        self.trainer = Trainer()

        # location of generative model params if not part of Model
        self.checkpoint = None

        # set parameters for graph
        self.graph = None
        self.saver = None
        self.merge_summaries = None
        self.init = None
        self.sess_config = tf.ConfigProto(device_count={'GPU': 1})  # gpu def
        self.np_seed = np_seed
        self.tf_seed = tf_seed

        # save constructor inputs for easy save/load
        constructor_inputs = {
            'inf_network': inf_network,
            'inf_network_params': inf_network_params,
            'gen_model': gen_model,
            'gen_model_params': gen_model_params,
            'np_seed': np_seed,
            'tf_seed': tf_seed}
        self.constructor_inputs = constructor_inputs

    def build_graph(self):
        """Build tensorflow computation graph for model"""
        raise NotImplementedError

    def _define_objective(self):
        """Objective function used to optimize model parameters"""
        self.objective = None
        raise NotImplementedError

    def train(self, **kwargs):
        """
        Train model

        See Trainer.train for input options
        """

        self.trainer.train(self, **kwargs)

    def sample(
            self, ztype='prior', num_samples=1, seed=None,
            linear_predictors=None, checkpoint_file=None,
            z_in = None, y_in = None, gmm_params = None, mark_probs = None):
        """
        Generate samples from prior/posterior and model

        Args:
            ztype (str): distribution used for latent state samples
                'prior' | 'posterior'
            num_samples (int, optional)
            seed (int, optional): random seed for reproducibly generating
                random samples
            linear_predictors (list of np arrays):
                1 x num_time_pts x dim_pred;
                there will be num_samples random samples for this one example
                of each linear predictor
            checkpoint_file (str, optional): checkpoint file specifying model
                from which to generate samples; if `None`, will then look for a
                checkpoint file created upon model initialization

        Returns:
            num_samples x num_time_pts x dim_obs numpy array: y
            num_samples x num_time_pts x dim_latent numpy array: z

        Raises:
            ValueError: for incorrect `ztype` values

        """

        self._check_graph()
        
        # intialize session
        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
#            print(self.gen_net.networks[0].layers[0].kernel.eval(sess))
            if z_in is not None:
                # this only works in one specific case
                ztmp = tf.convert_to_tensor(z_in, dtype = self.dtype)
                ymean = self.gen_net.networks[0].apply_network(ztmp)
                if mark_probs is not None:
#                    if mark_probs is None:
#                        raise ValueError('Must provide mark probabilities to evaluate this model')
                    F = tf.expand_dims(ymean, axis = 2)
                    W = tf.convert_to_tensor(mark_probs, dtype = self.dtype)
                    ymean = tf.squeeze(tf.matmul(F, W))
                
                y_ = tf.random_poisson(lam=ymean, shape=[1], dtype=tf.float32, seed = seed)
                y = np.squeeze(y_.eval())
                z = z_in
#                [y, z] = sess.run([y_, ztmp], feed_dict=feed_dict)
                
            elif ztype is 'prior':
                y, z = self.gen_net.sample(
                    sess, num_samples, seed, linear_predictors, mark_probs)

            elif ztype is 'posterior':
                if y_in is None:
                    raise ValueError('`y_in` required to sample from posterior')
#                 TODO: needs input to inference network as well
                z = self.inf_net.sample(sess, y_in, mark_probs = mark_probs, seed = seed)
                ztmp = tf.convert_to_tensor(z, dtype = self.dtype)
                ymean = self.gen_net.networks[0].apply_network(ztmp)
                if mark_probs is not None:
#                    if mark_probs is None:
#                        raise ValueError('Must provide mark probabilities to evaluate this model')
                    F = tf.transpose(ymean, (0, 2, 1, 3))
                    W = tf.convert_to_tensor(mark_probs, dtype = self.dtype)
                    ymean = tf.transpose(tf.matmul(F, W), (0, 2, 1, 3))
                    
                y_ = tf.random_poisson(lam=ymean, shape=[1], dtype=tf.float32, seed = seed)
                y = np.squeeze(y_.eval())
            else:
                raise ValueError('Invalid string "%s" for ztype argument')
                
            # simulate the marks -- takes a long time
            if gmm_params is not None:
                ztmp = tf.convert_to_tensor(z, dtype = self.dtype)
                f_k = self.gen_net.networks[0].apply_network(ztmp)
                f_k = f_k.eval()
                f_k = np.reshape(f_k,(-1,f_k.shape[2]))
                y_flat = np.reshape(y,(-1,self.dim_obs[0]))
                
                mrk = []
                ii = 0
                for i in range(self.dim_obs[0]):
                    mus = gmm_params[i][0]
                    sigs = gmm_params[i][1]
                    pi = gmm_params[i][2]
                    
                    dim_marks = mus.shape[0]
                    
                    k = pi.shape[0]
                    f_ik = f_k[:,ii:(ii+k)]
                    ii += k
                    
                    log_pi_hat = np.log(f_ik) + np.log(pi + 1e-16)[np.newaxis,:]
                    c = log_pi_hat.max(1)[:,np.newaxis]
                    log_norm = c + np.log(np.sum(np.exp(log_pi_hat - c), axis = 1))[:,np.newaxis]
                    log_pi_hat -= log_norm
                    pi_hat = np.exp(log_pi_hat)
                    
                    temp_mrk = np.zeros((0, dim_marks))
                    ind = np.zeros(0)
                    for t in np.where(y_flat[:,i] > 0)[0]:
                        Nt = int(y_flat[t,i])
                        zeta = rnd.multinomial(1, pi_hat[t,:], size = Nt).argmax(1)
                        mu = mus[:,zeta]
                        C = sigs[:,:,zeta]

                        m_t = np.array([rnd.multivariate_normal(mu[:,jj],C[:,:,jj]) \
                                        for jj in range(Nt)], dtype = np.float32)
                        temp_mrk = np.append(temp_mrk,m_t, axis = 0)
                        ind = np.append(ind, np.ones(Nt)*int(t))
                        
                    mrk.append([temp_mrk, ind])
            else:
                mrk = None
                
        return y, z, mrk
    
    def joint_intensity(
            self, values, ztype='prior', seed=None, checkpoint_file=None, 
            y_in = None, gmm_params = None):
        '''
        Evaluate the joint intensity (lambda) at specified values of the latent
        and (optionally) the marks.
        '''
        
        self._check_graph()
        
        # intialize session
        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
#            print(self.gen_net.networks[0].layers[0].kernel.eval())
            if gmm_params is None:
                ztmp = tf.convert_to_tensor(values, dtype = self.dtype)
                ymean = self.gen_net.networks[0].apply_network(ztmp)
                intensity = ymean.eval()
            else:
                ztmp = tf.convert_to_tensor(values[:,:self.dim_latent[0]], dtype = self.dtype)
                f_k = self.gen_net.networks[0].apply_network(ztmp)
                f_k = f_k.eval()
                f_k = np.reshape(f_k,(-1, self.gen_net.num_clusters))
                
                intensity = np.zeros((f_k.shape[0], self.dim_obs[0]))
                
                ii = 0
                for i in range(self.dim_obs[0]):
                    mus = gmm_params[i][0]
                    sigs = gmm_params[i][1]
                    pi = gmm_params[i][2]
                    
                    K = pi.shape[0]
                    f_ik = f_k[:,ii:(ii+K)]
                    ii += K
                    if values.shape[-1] > self.dim_latent[0]: # full joint
                        mrks = values[:,self.dim_latent[0]:]
                        probs = np.array([pi[k]*sts.multivariate_normal.pdf(mrks,mus[:,k],sigs[:,:,k]) \
                                          for k in range(K)]).T
#                        log_probs = np.log(probs)
#                        c = log_probs.max(1)[:,np.newaxis]
#                        log_norm = c + np.log(np.sum(np.exp(log_probs - c), axis = 1))[:,np.newaxis]
#                        log_probs -= log_norm
#                        probs = np.exp(log_probs)
                    else: # marginal joint
                        probs = pi[np.newaxis,:]
                        
                    intensity[:,i] = (probs*f_ik).sum(1)
                    
        return intensity

    def checkpoint_model(
            self, sess=None, checkpoint_file=None, save_filepath=False,
            print_filepath=False, opt_params=None):
        """
        Checkpoint model parameters in tf.Variables. The tensorflow graph will
        be constructed if necessary.

        Args:
            sess (tf.Session object, optional): current session object to run
                graph; if `None`, a session will be created
            checkpoint_file (str, optional): full path to output file; if
                `None`, the code will check for the `checkpoint_file` attribute
                of the model
            save_filepath (str, optional): save filepath as an attribute of the
                model
            print_filepath (bool, optional): print path of checkpoint file
            opt_params (dict): specify optimizer params if building graph for
                the first time

        Raises:
            ValueError: if no checkpoint_file is found

        """

        if checkpoint_file is None:
            if self.checkpoint is not None:
                checkpoint_file = self.checkpoint
            else:
                raise ValueError('Must specify checkpoint file')

        if not os.path.isdir(os.path.dirname(checkpoint_file)):
            os.makedirs(os.path.dirname(checkpoint_file))

        if sess is None:
            # assume we are saving an initial model
            # build tensorflow computation graph
            if self.graph is None:
                if opt_params is not None:
                    self.trainer.parse_optimizer_options(**opt_params)
                self.build_graph()

            # intialize session
            with tf.Session(graph=self.graph, config=self.sess_config) as sess:
                sess.run(self.init)
                # save graph
                tf.summary.FileWriter(checkpoint_file, graph=sess.graph)
                # checkpoint variables
                self.checkpoint_model(
                    sess=sess,
                    checkpoint_file=checkpoint_file,
                    save_filepath=False,
                    print_filepath=False)
                # save/print filepath in outer loop
                save_filepath = True
        
        else:
            self.saver.save(sess, checkpoint_file)

        if save_filepath:
            self.checkpoint = checkpoint_file

        if print_filepath:
            print('model checkpointed to %s' % checkpoint_file)

    def restore_model(self, sess, checkpoint_file=None):
        """
        Restore previously checkpointed model parameters in tf.Variables

        Args:
            sess (tf.Session object): current session object to run graph
            checkpoint_file (str): full path to saved model

        Raises:
            ValueError: If `checkpoint_file` is not a valid filename

        """

        if checkpoint_file is None:
            if self.checkpoint is not None:
                checkpoint_file = self.checkpoint
            else:
                raise ValueError('Must specify checkpoint file')

        if not os.path.isfile(checkpoint_file + '.meta'):
            raise ValueError(
                str('"%s" is not a valid filename' % checkpoint_file))

        # restore saved variables into tf Variables
        self.saver.restore(sess, checkpoint_file)

    def save_model(self, save_file):
        """
        Save constructor inputs of model using pickle

        Args:
            save_file (str): full path to output file

        Example:
            model_0 = Model(...) # call constructor
            model_0.train(...)   # should checkpoint models here
            model_0.save_model('/path/to/file/model_0')

            model_1 = Model.load_model('/path/to/file/model_0')

            In order for model_1 to use the parameters learned during the call
            to model_0.train(), model_0 must checkpoint its parameters and
            store them in the `checkpoint` Model attribute; this attribute will
            be used by model_1 if not `None` for restoring model parameters
            (see the Model.train function for more info on how to specify when
            model parameters are checkpointed)

        """

        import pickle

        if not os.path.isdir(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))

        # grab constructor inputs (along with model class specification)
        constructor_inputs = dict(self.constructor_inputs)

        # trainer parameters (for rebuilding graph)
        constructor_inputs['learning_alg'] = self.trainer.learning_alg
        constructor_inputs['opt_params'] = self.trainer.opt_params

        # save checkpoint file as well
        if self.checkpoint is not None:
            constructor_inputs['checkpoint_file'] = self.checkpoint
        else:
            print('warning: model has not been checkpointed; restoring this '
                  'model will result in random parameters')
            constructor_inputs['checkpoint_file'] = None

        with open(save_file, 'wb') as f:
            pickle.dump(constructor_inputs, f)

        print('model pickled to %s' % save_file)

    @classmethod
    def load_model(cls, save_file):
        """
        Restore previously saved Model object

        Args:
            save_file (str): full path to saved model

        Raises:
            ValueError: If `save_file` is not a valid filename

        """

        import pickle

        if not os.path.isfile(save_file):
            raise ValueError(str('%s is not a valid filename' % save_file))

        with open(save_file, 'rb') as f:
            constructor_inputs = pickle.load(f)

        print('model loaded from %s' % save_file)

        # extract model class to use as constructor
        model_class = constructor_inputs['model_class']
        del constructor_inputs['model_class']

        # extract trainer info for building graph
        learning_alg = constructor_inputs['learning_alg']
        del constructor_inputs['learning_alg']
        opt_params = constructor_inputs['opt_params']
        del constructor_inputs['opt_params']

        # tell model where to find checkpoint file for restoring parameters
        checkpoint_file = constructor_inputs['checkpoint_file']
        del constructor_inputs['checkpoint_file']
        if checkpoint_file is None:
            print('warning: model has not been checkpointed; restoring this '
                  'model will result in random parameters')

        # initialize model
        model = model_class(**constructor_inputs)
        model.checkpoint = checkpoint_file

        # specify trainer params
        model.trainer.parse_optimizer_options(
            learning_alg=learning_alg, **opt_params)

        # build graph
        model.build_graph()

        return model

    def _check_graph(self):
        if self.graph is None:
            self.build_graph()


class DynamicalModel(Model):
    """
    Models with dynamical generative models; should be subclassed by a specific
    model that implements a `build_graph` method
    """

    def __init__(
            self, inf_network=None, inf_network_params=None, gen_model=None,
            gen_model_params=None, np_seed=0, tf_seed=0):
        """
        Constructor for full Model; combines an inference network with a
        generative model and provides training functions

        Args:
            inf_network (InferenceNetwork class)
            inf_network_params (dict)
            gen_model (GenerativeModel class)
            gen_model_params (dict)
            np_seed (int)
            tf_seed (int)

        """

        super().__init__(
            inf_network=inf_network, inf_network_params=inf_network_params,
            gen_model=gen_model, gen_model_params=gen_model_params,
            np_seed=np_seed, tf_seed=tf_seed)

        # to clean up training functions
        self.dim_obs = self.gen_net.dim_obs
        self.dim_latent = self.gen_net.dim_latent
        self.num_time_pts = self.gen_net.num_time_pts

        # observations
        self.y_true = []

    def build_graph(self):
        """Build tensorflow computation graph for model"""
        raise NotImplementedError

    def _define_objective(self):
        """
        Objective function used to optimize model parameters

        This function uses the log-joint/entropy formulation of the ELBO
        """

        # expected value of log joint distribution
        with tf.variable_scope('log_joint'):
            self.log_joint = self.gen_net.log_density(
                self.y_true, self.inf_net.post_z_samples)

        # entropy of approximate posterior
        with tf.variable_scope('entropy'):
            self.entropy = self.inf_net.entropy()

        # objective to minimize
        self.objective = -self.log_joint - self.entropy

        # save summaries
        # with tf.variable_scope('summaries'):
        tf.summary.scalar('log_joint', self.log_joint)
        tf.summary.scalar('entropy', self.entropy)
        tf.summary.scalar('elbo', -self.objective)

    def get_dynamics_params(self, checkpoint_file=None):
        """
        Get parameters of generative model

        Args:
            checkpoint_file (str, optional): location of checkpoint file;
                if `None`, will then look for a checkpoint file created upon
                model initialization

        Returns:
            params (dict)

        """

        self._check_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
            params = self.gen_net.get_params(sess)

        return params

    def get_linear_params(self, checkpoint_file=None):
        """
        Get parameters of linear regressors

        Args:
            checkpoint_file (str, optional): location of checkpoint file;
                if `None`, will then look for a checkpoint file created upon
                model initialization

        Returns:
            params (dict)

        """

        self._check_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
            params = self.gen_net.get_linear_params(sess)

        return params

    def get_posterior_means(self, input_data=None, checkpoint_file=None, mark_probs = None):
        """
        Get posterior means from inference network

        Args:
            input_data (num_samples x num_time_pts x dim_obs tf.Tensor):
                data on which to condition the posterior means
            checkpoint_file (str, optional): location of checkpoint file
                specifying model from which to generate samples; if `None`,
                will then look for a checkpoint file created upon model
                initialization

        Returns:
            posterior_means (num_samples x num_time_pts x dim_latent tf.Tensor)

        """

        self._check_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
            posterior_means = self.inf_net.get_posterior_means(
                sess, input_data, mark_probs = mark_probs)

        return posterior_means

    def get_cost(self, observations=None, input_data=None, indxs=None,
                 checkpoint_file=None):
        """
        User function for retrieving cost

        Args:
            observations (num_samples x num_time_pts x dim_obs tf.Tensor):
                observations on which to condition the posterior means
            input_data (num_samples x num_time_pts x dim_input tf.Tensor,
                optional)
            indxs (list, optional): list of indices into observations and
                input_data
            checkpoint_file (str, optional): location of checkpoint file
                specifying model from which to generate samples; if `None`,
                will then look for a checkpoint file created upon model
                initialization

        Returns:
            float: value of objective function

        """

        if indxs is None:
            indxs = list(range(observations.shape[0]))

        self._check_graph()

        with tf.Session(graph=self.graph, config=self.sess_config) as sess:
            self.restore_model(sess, checkpoint_file=checkpoint_file)
            cost = self.trainer._get_cost(
                sess=sess, observations=observations, input_data=input_data,
                indxs=indxs)

        return cost


class LDSModel(DynamicalModel):
    """LDS generative model, various options for approximate posterior"""

    def __init__(
            self, inf_network=None, inf_network_params=None, gen_model=None,
            gen_model_params=None, couple_params=True, np_seed=0, tf_seed=0):
        """
        Constructor for full Model; see DynamicalModel for arg documentation

        Args:
            inf_network (InferenceNetwork class):
                MeanFieldGaussian | MeanFieldGaussianTemporal | SmoothingLDS
            inf_network_params (dict): see constructors in inference.py
            gen_model (GenerativeModel class)
                NetFLDS | NetLDS | FLDS | LDS
            gen_model_params (dict): see constructors in generative.py
            couple_params (bool): couple dynamical parameters of generative
                model and approximate posterior; only used when
                inf_network=SmoothingLDS and gen_model=*LDS
            np_seed (int): for training minibatches
            tf_seed (int): for initializing tf.Variables (sampling functions
                have their own seed arguments)
        """

        super().__init__(
            inf_network=inf_network, inf_network_params=inf_network_params,
            gen_model=gen_model, gen_model_params=gen_model_params,
            np_seed=np_seed, tf_seed=tf_seed)
        self.couple_params = couple_params

        self.constructor_inputs['model_class'] = LDSModel
        self.constructor_inputs['couple_params'] = couple_params

        self.obs_indxs = []

    def build_graph(self, opt_params=None):
        """Build tensorflow computation graph for model"""

        self.graph = tf.Graph()  # must be initialized before graph creation

        # build model graph
        with self.graph.as_default():

            # set random seed for this graph
            tf.set_random_seed(self.tf_seed)

            # construct data pipeline - assume that all data comes in as a
            # large block, and slice up tensor here to correspond to data from
            # distinct populations
            # NOTE: requires that `dim_obs` input to generative model
            # constructor is in same order as the data
            with tf.variable_scope('observations'):
                # one placeholder for all data
                self.y_true_ph = tf.placeholder(
                    dtype=self.dtype,
                    shape=[None, self.num_time_pts, sum(self.dim_obs)],
                    name='outputs_ph')
                # carve up placeholder into distinct populations
                indx_start = 0
                for pop, pop_dim in enumerate(self.dim_obs):
                    indx_end = indx_start + pop_dim
                    self.obs_indxs.append(
                        np.arange(indx_start, indx_end + 1, dtype=np.int32))
                    self.y_true.append(
                        self.y_true_ph[:, :, indx_start:indx_end])
                    indx_start = indx_end

            if self.couple_params:

                with tf.variable_scope('shared_vars'):
                    param_dict = self.gen_net.initialize_prior_vars()

                with tf.variable_scope('inference_network'):
                    self.inf_net.build_graph(param_dict)

                with tf.variable_scope('generative_model'):
                    self.gen_net.build_graph(
                        self.inf_net.post_z_samples, param_dict)

            else:

                with tf.variable_scope('inference_network'):
                    with tf.variable_scope('model_params'):
                        param_dict = self.gen_net.initialize_prior_vars()
                    self.inf_net.build_graph(param_dict)

                with tf.variable_scope('generative_model'):
                    with tf.variable_scope('model_params'):
                        param_dict = self.gen_net.initialize_prior_vars()
                    self.gen_net.build_graph(
                        self.inf_net.post_z_samples, param_dict)

            with tf.variable_scope('objective'):
                self._define_objective()

            with tf.variable_scope('optimizer'):
                self.trainer._define_optimizer_op(self)

            # add additional ops
            # for saving and restoring models (initialized after var creation)
            self.saver = tf.train.Saver()
            # collect all summaries into a single op
            self.merge_summaries = tf.summary.merge_all()
            # add variable initialization op to graph
            self.init = tf.global_variables_initializer()
