# Asymmetric Backpropagation, A Biological Plausible Network

Back propagation, which is based on chain rule of derivatives, is mathematically sound, but not biologically.  
The gradient, viewed in math, is very natural following the gradient formula. However, when some people want to view
it as biological feedback signal, things become implausible:   

To back propagate the signal in ordinary mathematical way, the weight is involved in the gradient expression. Some people
argue that the since neuron cell connection is not bidirectional, backward feedback can never know how strong a connection
is in the forward path, ie, weight in the forward path should not appear in the gradient function.

## Crazy Asymmetric Backpropagation Solution

Since we cannot use connection weight, some crazy person substitute the weight by a random feedback initialized similarly.  
Relevant Paper are:  
[1] How Important Is Weight Symmetry in Backpropagation? Qianli Liao, Joel Z. Leibo, Tomaso Poggio  
[2] Direct Feedback Alignment Provides Learning in Deep Neural Networks Arild Nøkland

NIPS paper on this, MIT Lab also on this. It seems that some people really want to play with it.

## Quick Summary Of Simple Math inside

Let h_i denote the activation value of layer i,  f denote the nonlinearity, J denote the loss function, and the dot circle denote the elementwise multiplication, B is the random feedback inserted:
![Math for asymmetric backpropagation](/Readme/asym.jpg)


## Goal of the Experiments

Here is a quick summary of the objectives in my experiments:
* Try asymmetric backpropagation, comparing it with original ones.
* See wether it works when layers goes deeper.
* How is multiple layer random feedback going?
* Is asymmetric backpropagation working when used in dense layers in convolutional neural network?
* Does untied convolution, which is more biological plausible than the original convolution, works?
* Is that possible to use random feedback in untied convolution layer?
* Is gradient clip helping to improve the training?
* Find evidence that updating the random feedback as the weight do can improve traininig.  
* Is there chance for asymmetric backpropagation to beat original one

## Some Quick Answers

* Deeper layer is more vulnerble
* Multiple asymmetric backpropagation is very unstable, especially for conv nets.
* It's possible to use random feedback in the last several dense layers in a conv network for classification, but the performance is poor.
* Untied convolution is pretty sound a new layer model, however the number of parameters are huge, which lead to severe over fitting.
* Random feedback in untied conv layer ruin the training.
* Gradient clip ruin all the training(for all clip value tried).
* Updating random feedback helps, though Prof. Amit argue that when you trained enough time results are similar. But I found that updating stabilize a lot of training cases.
* Random feedback never beat original ones, since this is a biological model rather than a mathematical model.

## Supporting Results
A summary of expriments performance can be found in \Readme\asymmetric_backpropagation.pdf
