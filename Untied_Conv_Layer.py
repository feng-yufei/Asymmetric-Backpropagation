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


def Untied_Conv_weight_test(shape,img_size,stride = 1):
    ipt_channel = shape[1]
    opt_channel = shape[0]
    filter_width = shape[2]
    n_filter = img_size - filter_width + 1
    weight =np.zeros([img_size*img_size*ipt_channel,n_filter*n_filter*opt_channel])
    for i in range(0,ipt_channel):
        for j in range(0,opt_channel):
            for x in range(0,n_filter):
                for y in range(0,n_filter):
                    for r in range(0,filter_width):
                        weight[i*img_size*img_size + (y+r)*img_size+ x + np.array(range(0, filter_width))  ,j*n_filter*n_filter + y*n_filter+ x ] = 1

    return weight



def conv_output_length(input_length, filter_size, stride, pad=0):

    if input_length is None:
        return None
    if pad == 'valid':
        output_length = input_length - filter_size + 1
    elif pad == 'full':
        output_length = input_length + filter_size - 1
    elif pad == 'same':
        output_length = input_length
    elif isinstance(pad, int):
        output_length = input_length + 2 * pad - filter_size + 1
    else:
        raise ValueError('Invalid pad: {0}'.format(pad))

    # This is the integer arithmetic equivalent to
    # np.ceil(output_length / stride)
    output_length = (output_length + stride - 1) // stride

    return output_length

class Fix_Conv2DLayer(Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 convolution=T.nnet.conv2d, **kwargs):
        super(Fix_Conv2DLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = as_tuple(filter_size, 2)
        self.stride = as_tuple(stride, 2)
        self.untie_biases = untie_biases
        self.convolution = convolution

        if pad == 'valid':
            self.pad = (0, 0)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = as_tuple(pad, 2, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W",trainable = False)
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters, self.output_shape[2], self.
                                output_shape[3])
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.input_shape[1]
        return (self.num_filters, num_input_channels, self.filter_size[0],
                self.filter_size[1])

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * 2

        output_rows = conv_output_length(input_shape[2],
                                         self.filter_size[0],
                                         self.stride[0],
                                         pad[0])

        output_columns = conv_output_length(input_shape[3],
                                            self.filter_size[1],
                                            self.stride[1],
                                            pad[1])

        return (input_shape[0], self.num_filters, output_rows, output_columns)

    def get_output_for(self, input, input_shape=None, **kwargs):
        # The optional input_shape argument is for when get_output_for is
        # called directly with a different shape than self.input_shape.
        if input_shape is None:
            input_shape = self.input_shape

        if self.stride == (1, 1) and self.pad == 'same':
            # simulate same convolution by cropping a full convolution
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode='full')
            shift_x = (self.filter_size[0] - 1) // 2
            shift_y = (self.filter_size[1] - 1) // 2
            conved = conved[:, :, shift_x:input.shape[2] + shift_x,
                            shift_y:input.shape[3] + shift_y]
        else:
            # no padding needed, or explicit padding of input needed
            if self.pad == 'full':
                border_mode = 'full'
                pad = [(0, 0), (0, 0)]
            elif self.pad == 'same':
                border_mode = 'valid'
                pad = [(self.filter_size[0] // 2,
                        (self.filter_size[0] - 1) // 2),
                       (self.filter_size[1] // 2,
                        (self.filter_size[1] - 1) // 2)]
            else:
                border_mode = 'valid'
                pad = [(self.pad[0], self.pad[0]), (self.pad[1], self.pad[1])]
            if pad != [(0, 0), (0, 0)]:
                input = padding.pad(input, pad, batch_ndim=2)
                input_shape = (input_shape[0], input_shape[1],
                               None if input_shape[2] is None else
                               input_shape[2] + pad[0][0] + pad[0][1],
                               None if input_shape[3] is None else
                               input_shape[3] + pad[1][0] + pad[1][1])
            conved = self.convolution(input, self.W, subsample=self.stride,
                                      image_shape=input_shape,
                                      filter_shape=self.get_W_shape(),
                                      border_mode=border_mode)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + self.b.dimshuffle('x', 0, 1, 2)
        else:
            activation = conved + self.b.dimshuffle('x', 0, 'x', 'x')

        return self.nonlinearity(activation)


