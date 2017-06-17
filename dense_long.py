from __future__ import print_function
import sys
import time
import theano
import theano.tensor as T
import theano.tensor.extra_ops as Ex
import lasagne
import load
import mini_batch
import numpy
import Untied_Conv_Layer as uconv
import SDenseLayer

numpy.random.seed(25)
def build_cnn(input_var):


    l_in = lasagne.layers.InputLayer(shape=(None,3072),
                                     input_var=input_var)

    l_hid_0  =lasagne.layers.DenseLayer(l_in,num_units=512,nonlinearity=lasagne.nonlinearities.rectify,
                                       b = None)

    l_hid_1 = lasagne.layers.DenseLayer(
    #l_hid_1 = SDenseLayer.SDenseLayer(
            l_hid_0,num_units = 512,
            nonlinearity=lasagne.nonlinearities.rectify,b = None)

    l_hid_2 = lasagne.layers.DenseLayer(
    #l_hid_2 = SDenseLayer.SDenseLayer(
            l_hid_1,num_units = 256,
           nonlinearity=lasagne.nonlinearities.rectify,b = None)

    #l_hid_3 = lasagne.layers.DenseLayer(
    l_hid_3 = SDenseLayer.SDenseLayer(
            l_hid_2,num_units = 256,
            nonlinearity=lasagne.nonlinearities.rectify,b = None)

    #l_out = lasagne.layers.DenseLayer(
    l_out = SDenseLayer.SDenseLayer(

            l_hid_3, num_units=10,
            nonlinearity=lasagne.nonlinearities.identity,b = None)

    return l_out



print("Loading data...")
# X_train, y_train, X_val, y_val,X_test,y_test= load.load_minst()
X_train, y_train, X_val, y_val = load.load_cifar10()
y_train = y_train.astype(numpy.int32)
y_val = y_val.astype(numpy.int32)
X_test = X_val
y_test = y_val
pass_weight = None

input_var = T.fmatrix('inputs')
target_var = T.ivector('targets')

print("Building model and...")

network = build_cnn(input_var)
prediction = lasagne.layers.get_output(network)
y1_hot = Ex.to_one_hot(target_var,10)
loss = T.mean(T.mul(y1_hot, T.nnet.relu(1 - prediction )) + 1.0/9*T.mul(1 - y1_hot, T.nnet.relu(1 + prediction )))
#loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
#loss = loss.mean()

# Get network params, with specifications of manually updated ones
params = lasagne.layers.get_all_params(network, trainable=True)
#updates = lasagne.updates.sgd(loss,params,learning_rate=0.01)
updates = lasagne.updates.adam(loss,params)
#updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.01)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
y1_hot_t = Ex.to_one_hot(target_var,10)
test_loss = T.mean(T.mul(y1_hot_t, T.nnet.relu(1 - test_prediction)) + 1.0/9*T.mul(1 - y1_hot_t, T.nnet.relu(1 + test_prediction)))
#test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
#test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

# Compile theano function computing the training validation loss and accuracy:
train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


# The training loop
print("Starting training...")
num_epochs = 200
for epoch in range(num_epochs):

    # In each epoch, we do a full pass over the training data:

    train_err = 0
    train_batches = 0
    start_time = time.time()

    for batch in mini_batch.iterate_minibatches(X_train, y_train, 500, shuffle=True):

        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1


    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0

    for batch in mini_batch.iterate_minibatches(X_val, y_val, 500, shuffle=False):

        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1


        # Then we print the results for this epoch:

    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))

    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

"""
params = lasagne.layers.get_all_params(network, trainable=True)
W1 = params[0].get_value()
W2 = params[1].get_value()
W3 = params[2].get_value()
W1 = uconv.Untied_Conv_weight_convert(W1,32)
W2 = uconv.Untied_Conv_weight_convert(W2,16)
W3 = uconv.Untied_Conv_weight_convert(W3,6)
numpy.savez('C:\Users\zjufe\PycharmProjects\\neural\Save\conv_1_weight.npz',W1 = W1)
numpy.savez('C:\Users\zjufe\PycharmProjects\\neural\Save\conv_2_weight.npz',W2 = W2)
numpy.savez('C:\Users\zjufe\PycharmProjects\\neural\Save\conv_3_weight.npz',W3 = W3)
print("::Saved::")
"""



