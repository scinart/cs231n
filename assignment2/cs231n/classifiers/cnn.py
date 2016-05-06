import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    std = weight_scale
    C,H,W=input_dim[0],input_dim[1],input_dim[2]
    self.params['W1'] = std * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = std * np.random.randn(num_filters*W*H/2/2, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = std * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out,cache1 = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
    out,cache2 = affine_relu_forward(out,W2,b2)
    scores,cache3 = affine_forward(out,W3,b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss(scores,y)
    loss += 0.5*self.reg*np.sum(self.params['W3']*self.params['W3'])
    loss += 0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])
    loss += 0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])
    dout,grads['W3'],grads['b3']=affine_backward(dout,cache3)
    dout,grads['W2'],grads['b2']=affine_relu_backward(dout,cache2)
    dout,grads['W1'],grads['b1']=conv_relu_pool_backward(dout,cache1)
    grads['W3']+=self.reg*self.params['W3']
    grads['W2']+=self.reg*self.params['W2']
    grads['W1']+=self.reg*self.params['W1']

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass




class MyConvNet(object):

  def __init__(self, input_dim=(3, 32, 32), meta_params={}, weight_scale=1e-3, reg=0.0, dtype=np.float32):
    #             conv (F,size,stride,pad)
    # (N,C,H,W)----------------------------->(N,F,(H+2*pad-size)/stride+1,(W+2*pad-size)/stride+1)

    #             pool (h,w,stride)
    # (N,C,H,W)----------------------------->(N,F,(H-h)/stride+1,(W-w)/stride+1)

    #             affine (C*H*W,D)
    # (N,C,H,W)----------------------------->(N,D)

    # useful layers
    # cv conv              (F,size,stride,pad)
    # cp conv+pool         (F,size,stride,pad,h,w,pool_stride)
    # af affine            (C*H*W,D)
    # sm softmax           (C*H*W, num_class)

    self.meta_params = {}
    self.params = {}
    self.bn_params = {}
    self.reg = reg
    self.dtype = dtype

    num_layers = len(meta_params)
    self.meta_params['num_layers'] = num_layers
    # for k, v in meta_params.iteritems():
    #   print k,v

    # rename parameter.
    std = weight_scale

    prev_dim=(3, 32, 32)
    for i in range(1,num_layers+1):
      print 'DEBUG: prev_dim',prev_dim
      layer=meta_params[i-1]
      self.meta_params['layer'+str(i)]=layer['type']
      if(layer['type']=='cp'):
        self.params['W'+str(i)] = std * np.random.randn(layer['F'], prev_dim[0], layer['size'], layer['size'])
        self.params['b'+str(i)] = np.zeros(layer['F'])
        self.meta_params['conv_param'+str(i)] = {'stride': layer['stride'], 'pad': layer['pad']}
        self.meta_params['pool_param'+str(i)] = {'pool_height': layer['h'], 'pool_width': layer['w'], 'stride': layer['pool_stride']}
        hh = (prev_dim[1]+2*layer['pad']-layer['size'])/layer['stride']+1
        ww = (prev_dim[2]+2*layer['pad']-layer['size'])/layer['stride']+1
        hhh= (hh-layer['h'])/layer['pool_stride']+1
        www= (ww-layer['w'])/layer['pool_stride']+1
        prev_dim=(layer['F'], hhh, www)
      elif(layer['type']=='bn' or layer['type']=='sbn'):
        self.params['gamma'+str(i)]=np.ones(prev_dim[0])
        self.params['beta'+str(i)]=np.ones(prev_dim[0])
        self.bn_params['bn_params'+str(i)]={}
      elif(layer['type']=='cv'):
        self.params['W'+str(i)] = std * np.random.randn(layer['F'], prev_dim[0], layer['size'], layer['size'])
        self.params['b'+str(i)] = np.zeros(layer['F'])
        self.meta_params['conv_param'+str(i)] = {'stride': layer['stride'], 'pad': layer['pad']}
        hh = (prev_dim[1]+2*layer['pad']-layer['size'])/layer['stride']+1
        ww = (prev_dim[2]+2*layer['pad']-layer['size'])/layer['stride']+1
        prev_dim=(layer['F'], hh, ww)
      elif (layer['type']=='af'):
        self.params['W'+str(i)] = std * np.random.randn(np.prod(prev_dim), layer['D'])
        self.params['b'+str(i)] = np.zeros(layer['D'])
        prev_dim=(layer['D'],)
      elif (layer['type']=='sm'):
        self.params['W'+str(i)] = std * np.random.randn(np.prod(prev_dim), layer['num_class'])
        self.params['b'+str(i)] = np.zeros(layer['num_class'])

    for k, v in self.params.iteritems():
      print 'DEBUG:', k, v.shape

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):

    mids = {}
    out = None
    out = X

    mode = 'test' if y is None else 'train'

    for k, v in self.bn_params.iteritems():
      self.bn_params[k]['mode']=mode

    for i in range(1,self.meta_params['num_layers']+1):
      if(self.meta_params['layer'+str(i)] == 'cv'):
        out,mids['cache'+str(i)] = conv_relu_forward(out,self.params['W'+str(i)],self.params['b'+str(i)],self.meta_params['conv_param'+str(i)])
      elif(self.meta_params['layer'+str(i)] == 'cp'):
        out,mids['cache'+str(i)] = conv_relu_pool_forward(out,self.params['W'+str(i)],self.params['b'+str(i)],self.meta_params['conv_param'+str(i)],self.meta_params['pool_param'+str(i)])
      elif(self.meta_params['layer'+str(i)] == 'sbn'):
        out,mids['cache'+str(i)] = spatial_batchnorm_forward(out,self.params['gamma'+str(i)],self.params['beta'+str(i)],self.bn_params['bn_params'+str(i)])
      elif(self.meta_params['layer'+str(i)] == 'bn'):
        out,mids['cache'+str(i)] = batchnorm_forward(out,self.params['gamma'+str(i)],self.params['beta'+str(i)],self.bn_params['bn_params'+str(i)])
      elif(self.meta_params['layer'+str(i)]=='af'):
        out,mids['cache'+str(i)] = affine_relu_forward(out,self.params['W'+str(i)],self.params['b'+str(i)])
      elif(self.meta_params['layer'+str(i)]=='sm'):
        out,mids['cache'+str(i)] = affine_forward(out,self.params['W'+str(i)],self.params['b'+str(i)])

    scores = out
    if y is None:
      return scores

    loss, grads = 0, {}
    loss, dout = softmax_loss(scores,y)

    # add l2 regularization
    for i in range(1,self.meta_params['num_layers']+1):
      if(self.meta_params['layer'+str(i)] != 'bn' and self.meta_params['layer'+str(i)] != 'sbn'):
        loss += 0.5*self.reg*np.sum(self.params['W'+str(i)]*self.params['W'+str(i)])

    for i in range(self.meta_params['num_layers'],0,-1):
      if(self.meta_params['layer'+str(i)] == 'cv'):
        dout,grads['W'+str(i)],grads['b'+str(i)]=conv_relu_backward(dout,mids['cache'+str(i)])
      elif(self.meta_params['layer'+str(i)] == 'sbn'):
        dout,grads['gamma'+str(i)],grads['beta'+str(i)]=spatial_batchnorm_backward(dout,mids['cache'+str(i)])
      elif(self.meta_params['layer'+str(i)] == 'bn'):
        dout,grads['gamma'+str(i)],grads['beta'+str(i)]=batchnorm_backward(dout,mids['cache'+str(i)])
      elif(self.meta_params['layer'+str(i)] == 'cp'):
        dout,grads['W'+str(i)],grads['b'+str(i)]=conv_relu_pool_backward(dout,mids['cache'+str(i)])
      elif(self.meta_params['layer'+str(i)] == 'af'):
        dout,grads['W'+str(i)],grads['b'+str(i)]=affine_relu_backward(dout,mids['cache'+str(i)])
      elif(self.meta_params['layer'+str(i)] == 'sm'):
        dout,grads['W'+str(i)],grads['b'+str(i)]=affine_backward(dout,mids['cache'+str(i)])

    # dout,grads['W3'],grads['b3']=affine_backward(dout,cache3)
    # dout,grads['W2'],grads['b2']=affine_relu_backward(dout,cache2)
    # dout,grads['W1'],grads['b1']=conv_relu_pool_backward(dout,cache1)
    # grads['W3']+=self.reg*self.params['W3']
    # grads['W2']+=self.reg*self.params['W2']
    # grads['W1']+=self.reg*self.params['W1']
    return loss, grads


pass
