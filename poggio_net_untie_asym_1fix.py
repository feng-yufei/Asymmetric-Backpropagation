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
import gradient_descent
import SDenseLayer

#print(theano.config.device)

numpy.random.seed(25)
def build_cnn(input_var,pass_weight):


    l_in = lasagne.layers.InputLayer(shape=(None,3,32,32),
                                     input_var=input_var)

    l_conv_1 = lasagne.layers.Conv2DLayer(
            l_in, num_filters=32, filter_size=(5, 5),pad = 'same',
            nonlinearity=lasagne.nonlinearities.rectify,W=pass_weight[0],b = None)
    
    l_conv_1.W = l_conv_1.add_param(pass_weight[0],l_conv_1.get_W_shape(),name = "W",trainable = False)

    l_pad_2 = lasagne.layers.PadLayer(l_conv_1, width=1)

    l_pool_1 = lasagne.layers.MaxPool2DLayer(l_pad_2, pool_size=(3, 3), stride=2)

    l_pad_3 = lasagne.layers.PadLayer(l_pool_1, width=2)

    l_reshape_2 = lasagne.layers.ReshapeLayer(l_pad_3, ([0], -1))

    #l_conv_2 = uconv.Untied_Conv_Layer(
    l_conv_2 = uconv.Untied_Conv_Layer_Random_Feedback(
        l_reshape_2,num_units=16*16*64,
        nonlinearity=lasagne.nonlinearities.tanh, W=pass_weight[1], b=None)

    l_back_2 = lasagne.layers.ReshapeLayer(l_conv_2, ([0], 64, 16, 16))

    l_pad_4 = lasagne.layers.PadLayer(l_back_2, width=1)

    l_pool_2 = lasagne.layers.Pool2DLayer(l_pad_4, pool_size=(3, 3), stride=2,
                                          mode="average_inc_pad")

    l_pad_5 = lasagne.layers.PadLayer(l_pool_2, width=2)

    l_reshape_3 = lasagne.layers.ReshapeLayer(l_pad_5, ([0], -1))

    #l_conv_3 = uconv.Untied_Conv_Layer(
    l_conv_3 = uconv.Untied_Conv_Layer_Random_Feedback(
        l_reshape_3, num_units=8*8*64,
        nonlinearity=lasagne.nonlinearities.tanh, W=pass_weight[2], b=None)

    l_back_3 = lasagne.layers.ReshapeLayer(l_conv_3, ([0], 64, 8, 8 ))

    l_pool_3 = lasagne.layers.Pool2DLayer(l_back_3, pool_size=(3, 3), stride=2, pad=(1, 1),
                                          mode="average_inc_pad")

    #l_hid_1 = lasagne.layers.DenseLayer(
    l_hid_1=SDenseLayer.SDenseLayer(
        l_pool_3, num_units=128,
        nonlinearity=lasagne.nonlinearities.tanh, b=None)

    #l_out = lasagne.layers.DenseLayer(
    l_out = SDenseLayer.SDenseLayer(
        l_hid_1, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax, b=None)

    return l_out



print("Loading data...")
# X_train, y_train, X_val, y_val,X_test,y_test= load.load_minst()
X_train, y_train, X_val, y_val = load.loadcifar10()
y_train = y_train.astype(numpy.int32)
y_val = y_val.astype(numpy.int32)
X_test = X_val
y_test = y_val

W1 = numpy.array(numpy.load('./neural/Save/conv_1_weight_tied.npz',)['W1']).astype(numpy.float32)
W2 = numpy.array(numpy.load('./neural/Save/conv_2_weight.npz',)['W2']).astype(numpy.float32)
W3 = numpy.array(numpy.load('./neural/Save/conv_3_weight.npz',)['W3']).astype(numpy.float32)
print("Convolutional Layer Weight Shapes: ")
print(W1.shape)
print(W2.shape)
print(W3.shape)
#mask_1 = theano.shared(numpy.array(1.0*(W1!=0)).astype(numpy.float32))
mask_2 = theano.shared(numpy.array(1.0*(W2!=0)).astype(numpy.float32))
mask_3 = theano.shared(numpy.array(1.0*(W3!=0)).astype(numpy.float32))
W1  =theano.shared(W1)
W2  =theano.shared(W2)
W3  =theano.shared(W3)
pass_weight = [W1,W2,W3]


input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

print("Building model and...")

network = build_cnn(input_var,pass_weight)
prediction = lasagne.layers.get_output(network)
#y1_hot = Ex.to_one_hot(target_var,10)
#loss = T.mean(T.mul(y1_hot, T.nnet.relu(1 - prediction )) + 1.0/9*T.mul(1 - y1_hot, T.nnet.relu(1 + prediction )))
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()

# Get network params, with specifications of manually updated ones
# updates = lasagne.updates.sgd(loss,params,learning_rate=0.01)
# updates = lasagne.updates.adam(loss,params)
# updates = lasagne.updates.nesterov_momentum(loss,params,learning_rate=0.01)
params = lasagne.layers.get_all_params(network, trainable=True)
spec_param =lasagne.layers.get_all_params(network,manual_update = True,rf = False)
rf_param = lasagne.layers.get_all_params(network,manual_update = True,rf = True)
print("Number of Untied Conv Params:{}".format(len(spec_param)))
print("Numver of Conv Random Feedback Possible for Updates:{}".format(len(rf_param)))
masks =[mask_2,mask_3]
#updates  = gradient_descent.sgd(loss,params,spec_param,masks,learning_rate=0.01)
updates  = gradient_descent.adam_update_rf(loss,params,spec_param,rf_param,masks)


test_prediction = lasagne.layers.get_output(network, deterministic=True)
y1_hot_t = Ex.to_one_hot(target_var,10)
#test_loss = T.mean(T.mul(y1_hot_t, T.nnet.relu(1 - test_prediction)) + 1.0/9*T.mul(1 - y1_hot_t, T.nnet.relu(1 + test_prediction)))
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)

# Compile theano function computing the training validation loss and accuracy:
train_fn = theano.function([input_var, target_var], loss, updates=updates)
val_fn = theano.function([input_var, target_var], [test_loss, test_acc])


# The training loop
print("Starting training...")
num_epochs = 100
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
W2 = uconv.Untied_Conv_weight_convert(W2,16)
W3 = uconv.Untied_Conv_weight_convert(W3,6)
numpy.savez('E:\PyCharm Project\RandBackProp\Save\conv_1_weight.npz',W1 = W1)
numpy.savez('E:\PyCharm Project\RandBackProp\Save\conv_2_weight.npz',W2 = W2)
numpy.savez('E:\PyCharm Project\RandBackProp\Save\conv_3_weight.npz',W3 = W3)
print("::Saved::")
"""
