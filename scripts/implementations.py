# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

#############################
#      HELPER FUNCTIONS     #
#############################
def compute_mse(y, tx, w):
    """Compute the loss by mse."""
    e = y - tx @ w
    return e.T.dot(e) / (2 * len(e))

def compute_gradient(y, tx, w):
    """Compute the gradient of MSE loss function"""
    n = y.shape[0]
    e = y - tx @ w
    return (-1./n) * np.dot(tx.T, e)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def sigmoid(t):
    """Apply the sigmoid function to t"""
    return 1./(1 + np.exp(-t))

def calculate_reg_log_loss(y, tx, w, lambda_):
    """Compute the loss: negative log likelihood."""
    loss = 0
    for i in range(len(y)):
        arg = tx[i].T @ w
        loss += np.log(1 + np.exp(arg)) - y[i] * arg

    return loss + (lambda_ / 2) * np.linalg.norm(w)

def calculate_reg_log_gradient(y, tx, w, lambda_):
    """Compute the gradient of log loss."""
    grad = tx.T @ (sigmoid(tx @ w) - y.reshape((y.shape[0], 1)))
    return grad.reshape((w.shape[0], 1)) + lambda_ * w





#############################
#  PROJECT IMPLEMENTATIONS  #
#############################


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent
    """
    w = initial_w
    loss = None

    for iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)

        w = w - gamma * grad

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent
    """
    w = initial_w
    loss = None

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_mse(minibatch_y, minibatch_tx, w)

            w = w - gamma * grad

    return w, loss

def least_squares(y, tx):
    """
    Least squares regression using normal equations
    """
    w_s = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_mse(y, tx, w_s)
    return w_s, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    """
    n = tx.shape[0]
    lambda_prime = 2 * n * lambda_

    w_s = np.linalg.solve(tx.T @ tx + lambda_prime * np.identity(tx.shape[1]), tx.T @ y)
    loss = compute_mse(y, tx, w_s) + lambda_ * np.dot(w_s, w_s)

    return w_s, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent or SGD
    """
    w = initial_w
    loss = None

    for iter in range(max_iters):
        grad = calculate_reg_log_gradient(y, tx, w, 0)
        loss = calculate_reg_log_loss(y, tx, w, 0)

        w = w - gamma * grad

    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent or SGD
    """
    w = initial_w
    loss = None

    for n_iter in range(max_iters):
        grad = calculate_reg_log_gradient(y, tx, w, lambda_)
        loss = calculate_reg_log_loss(y, tx, w, lambda_)

        w = w - gamma * grad

    return w, loss
