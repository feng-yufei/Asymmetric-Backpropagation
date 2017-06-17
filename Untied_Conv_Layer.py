import numpy as np
import theano.tensor as T
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers.base import Layer
from lasagne.utils import as_tuple
from lasagne.theano_extensions import conv, padding
import tensor_op



# untied convolution, each receptive square have its own weight
# enormous number of parameters, severe overfitting, however the weight sharing of
# different location is not biological plausible.
class Untied_Conv_Layer(Layer):

    def __init__(self, incoming, num_units, W,
                 b=None, nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(Untied_Conv_Layer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W",trainable= False,manual_update = True)
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        # The untied convolution is implemented by a dot operator
        # The weight matrix is extremely large, which is computational/memory inefficient.  
        activation = T.dot(input, self.W)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

    
# not only untied convolution, but also backpropagate random feed back 
# Asymmetric untied convolution
class Untied_Conv_Layer_Random_Feedback(Layer):

    def __init__(self, incoming, num_units, W,W_s = init.GlorotUniform(),
                 b=None, nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(Untied_Conv_Layer_Random_Feedback, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W",trainable= False,manual_update = True)
        self.W_s = self.add_param(W_s, (num_inputs, num_units), name="W_s", trainable=False, manual_update=True)
        mask = np.array(1.0 * (self.W.get_value() != 0)).astype(np.float32)
        self.W_s.set_value((self.W_s.get_value()*mask).astype(np.float32))
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        #activation = T.dot(input, self.W)
        activation = tensor_op.s_dot(input,self.W,self.W_s)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

    
# convert a weight of the original convolution to the very large very sparse matrix
# used in untied convolution
def Untied_Conv_weight_convert(tied_weight,img_size,stride = 1):
    ipt_channel = tied_weight.shape[1]
    opt_channel = tied_weight.shape[0]
    filter_width = tied_weight.shape[2]
    n_filter = img_size - filter_width + 1
    weight =np.zeros([img_size*img_size*ipt_channel,n_filter*n_filter*opt_channel])
    for i in range(0,ipt_channel):
        for j in range(0,opt_channel):
            for x in range(0,n_filter):
                for y in range(0,n_filter):
                    for r in range(0,filter_width):
                        weight[i*img_size*img_size + (y+r)*img_size+ x + np.array(range(0, filter_width))  ,j*n_filter*n_filter + y*n_filter+ x ] =\
                            np.transpose(tied_weight[j,i,r,:])
    return weight



