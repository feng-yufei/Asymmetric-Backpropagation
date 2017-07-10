from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T
from lasagne import utils

def get_or_compute_grads(loss_or_grads, params):
    if isinstance(loss_or_grads, list):
        if not len(loss_or_grads) == len(params):
            raise ValueError("Got %d gradient expressions for %d parameters" %
                             (len(loss_or_grads), len(params)))
        return loss_or_grads
    else:
        return theano.grad(loss_or_grads, params)


def sgd(loss_or_grads, params,special_params,masks, learning_rate):

    grads = get_or_compute_grads(loss_or_grads, params)
    all_spec_grads = get_or_compute_grads(loss_or_grads, special_params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

    for param, grad, mask in zip(special_params, all_spec_grads, masks):
        updates[param] = param - learning_rate * grad * mask*30

    return updates


def adam(loss_or_grads, params,special_params,masks, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    all_grads = get_or_compute_grads(loss_or_grads, params)
    all_spec_grads = get_or_compute_grads(loss_or_grads, special_params)
    t_prev = theano.shared(utils.floatX(0.))
    updates = OrderedDict()

    t = t_prev + 1
    a_t = learning_rate * T.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1 * m_prev + (1 - beta1) * g_t
        v_t = beta2 * v_prev + (1 - beta2) * g_t ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    for param, g_t,mask in zip(special_params, all_spec_grads, masks):

        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1 * m_prev + (1 - beta1) * (g_t*mask)
        v_t = beta2 * v_prev + (1 - beta2) * (g_t*mask) ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_prev] = t
    return updates

def adam_update_rf(loss_or_grads, params,special_params,rf,masks, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):
    all_grads = get_or_compute_grads(loss_or_grads, params)
    all_spec_grads = get_or_compute_grads(loss_or_grads, special_params)
    t_prev = theano.shared(utils.floatX(0.))
    updates = OrderedDict()

    t = t_prev + 1
    a_t = learning_rate * T.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

    for param, g_t in zip(params, all_grads):
        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1 * m_prev + (1 - beta1) * g_t
        v_t = beta2 * v_prev + (1 - beta2) * g_t ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    for param, rfs ,g_t, mask in zip(special_params,rf, all_spec_grads, masks):

        value = param.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=param.broadcastable)

        m_t = beta1 * m_prev + (1 - beta1) * (g_t*mask)
        v_t = beta2 * v_prev + (1 - beta2) * (g_t*mask) ** 2
        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step
        updates[rfs] = rfs - step

    updates[t_prev] = t
    return updates