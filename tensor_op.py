from theano import gof
import theano.tensor as T
import numpy


class SDotOp(gof.Op):

    __props__ = ("name", "fn")

    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

    def make_node(self, x, y,sub_mat):

        return gof.Apply(self, [x, y,sub_mat], [T.fmatrix()])

    def perform(self, node, inp, out):
        x, y,sub_mat = inp
        z, = out
        z[0] = self.fn(x,y)

    def __str__(self):
        return self.name

    def grad(self,inp, grads):
        x, y, sub_mat = inp
        gz, = grads

        x_grad = T.dot(gz,sub_mat.T)
        y_grad = T.dot(x.T, gz)
        #sub_grad = y_grad*0
        sub_grad = y_grad

        rval = x_grad, y_grad, sub_grad

        return rval


s_dot = SDotOp(name='SDot',
                     fn=lambda x, y: numpy.dot(x,y))




