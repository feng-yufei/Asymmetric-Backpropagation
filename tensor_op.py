from theano import gof
import theano.tensor as T
import numpy

# customized theano tensor operation, for asymmetric backpropagation
# Things are simple
# When we compute the gradient in the auto differentiation graph
# when computing gradient for previous hidden state, we need a dot involving gradient and the weight
# Now substitute the weight with something else, possibly random, and we can update the random feedback the same as the weight
# biological neural network is crazy things indeed


# This is customized API that theano provided.
# everything except the gradient is the same as theano.tensor.dot
class SDotOp(gof.Op):

    __props__ = ("name", "fn")

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def make_node(self, x, y,sub_mat):

        return gof.Apply(self, [x, y,sub_mat], [T.fmatrix()])

    def perform(self, node, inp, out):
        # three things involved: X,Y, random feedback sub_mat
        # sub_mat do not goes into the forward path
        x, y,sub_mat = inp
        z, = out
        z[0] = self.fn(x,y)

    def __str__(self):
        return self.name

    def grad(self,inp, grads):
        x, y, sub_mat = inp
        gz, = grads

        # gradient in the backpropagation is substituted here
        x_grad = T.dot(gz,sub_mat.T)
        y_grad = T.dot(x.T, gz)
        
        # update the random feedback, or not(force gradient to be 0 for random feedback)
        #sub_grad = y_grad*0
        sub_grad = y_grad

        rval = x_grad, y_grad, sub_grad

        return rval

# optional algorithm, will be automatically changed into GPU operation
s_dot = SDotOp(name='SDot',
                     fn=lambda x, y: numpy.dot(x,y))




