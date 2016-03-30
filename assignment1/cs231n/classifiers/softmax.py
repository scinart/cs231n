import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train,dim=X.shape
  num_class=W.shape[1]
  nc = X.dot(W)
  expnc = np.exp(nc)
  sum_nc = np.sum(expnc, axis=1)
  nc_correct = expnc[np.arange(num_train), y]
  loss -= np.sum(np.log(nc_correct/sum_nc)) / num_train

  for i in range(num_train):
    dW[:,y[i]] += -1 * X[i,:]
    # dW += (X[i]/sum_nc[i]).reshape(dim,1)
    for j in range(num_class):
      dW[:,j] += (X[i,:]*expnc[i,j]/sum_nc[i])
  dW /= num_train

  dW += reg * abs(W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train,dim=X.shape
  num_class=W.shape[1]
  nc = X.dot(W)
  expnc = np.exp(nc)
  sum_nc = np.sum(expnc, axis=1)
  nc_correct = expnc[np.arange(num_train), y]
  loss -= np.sum(np.log(nc_correct/sum_nc)) / num_train

  dW = X.T.dot(expnc / sum_nc.reshape(num_train,1))
  for i in range(num_train):
    dW[:,y[i]] += -1 * X[i,:]

  dW /= num_train

  dW += reg * abs(W)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

