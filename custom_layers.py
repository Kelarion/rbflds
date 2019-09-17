#%%

import tensorflow as tf
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.eager import context
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers


class RBFLayer(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 sigma = 1,
                 kernel_initializer ='glorot_uniform',
                 kernel_regularizer = None,
                 kernel_constraint = None,
                 activation = tf.exp,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 activity_regularizer=None,
                 **kwargs):
        '''
        A layer which applies and RBF to the input space, and allows for the 
        center to be learned as the weight matrix. Specifically, applies:
            f(x) = \exp{(W*x - (1/2)(||x||^2 + ||W||^2))/s}
        which is a way of re-writing the RBF equation by using:
            ||x - w||^2 = ||x||^2 + ||w||^2 - 2<w, x>
        '''
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(RBFLayer, self).__init__(
                activity_regularizer=regularizers.get(activity_regularizer), 
                **kwargs)
        self.units = units
        self.sigma = sigma
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.activation = activation
        
    def build(self, input_shape):
        self.kernel = self.add_variable(
                "kernel",
                shape = [int(input_shape[-1]), self.units],
                initializer = self.kernel_initializer,
                regularizer = self.kernel_regularizer,
                constraint = self.kernel_constraint)
    
    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)
        
        if rank > 2:
          # Broadcasting is required for the inputs.
          outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
          # Reshape the output back to the original ndim of the input.
          if not context.executing_eagerly():
            shape = inputs.get_shape().as_list()
            output_shape = shape[:-1] + [self.units]
            outputs.set_shape(output_shape)
        else:
          outputs = gen_math_ops.mat_mul(inputs, self.kernel)
          
        w_norm = 0.5*tf.linalg.norm(self.kernel, axis = 0)**2
        x_norm = tf.expand_dims(0.5*tf.linalg.norm(inputs, axis = -1)**2, axis = -1)
        
        outputs = gen_math_ops.add(outputs,-w_norm)
        outputs = gen_math_ops.add(outputs,-x_norm)
        outputs /= (self.sigma**2)
        
        return self.activation(outputs)
        
        