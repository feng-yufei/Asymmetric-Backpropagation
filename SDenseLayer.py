import numpy as np
import theano.tensor as T
from lasagne import init
from lasagne import nonlinearities
from lasagne.layers.base import Layer
import tensor_op



class SDenseLayer(Layer):

    def __init__(self, incoming, num_units, W=init.GlorotUniform(),W_s = init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
                 **kwargs):
        super(SDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        self.W_s = self.add_param(W_s, (num_inputs, num_units), name="W_s")
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

        activation = tensor_op.s_dot(input, self.W,self.W_s)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


