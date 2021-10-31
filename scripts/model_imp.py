import numpy as np
from matplotlib import pyplot as plt
from implementations import *

#############################
#   MODEL IMPLEMENTATIONS   #
#############################

# IDEA: implement ridge regression pipeline from ex4 (for parameter tuning) and then perhaps
# replace the actual ridge regression with logistic regression!


def reg_logistic_regression_mod(y, tx, lambda_, initial_w, max_iters, gamma, threshold):
    """
    Regularized logistic regression using gradient descent or SGD,
    stop early if difference of last two losses is smaller than threshold
    """
    w = initial_w
    losses = []

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, int(y.shape[0] / 10)):
            grad = calculate_reg_log_gradient(minibatch_y, minibatch_tx, w, lambda_)
            loss = calculate_reg_log_loss(minibatch_y, minibatch_tx, w, lambda_)

            w = w - gamma * grad

            losses.append(loss)
            if (len(losses) > 1 and np.abs(losses[-2] - losses[-1]) < threshold):
                return w, loss

    return w, loss


def split_data(x, y, ratio, seed):
    """
    Split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    np.random.seed(seed)

    tr_x = []
    tr_y = []
    te_x = []
    te_y = []
    for i in range(0, x.shape[0]):
        r = np.random.rand()

        if (r < ratio):
            tr_x.append(x[i])
            tr_y.append(y[i])
        else:
            te_x.append(x[i])
            te_y.append(y[i])

    return np.array(tr_x), np.array(tr_y), np.array(te_x), np.array(te_y)


def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold"""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, tx, k_indices, k, lambda_, degree):
    """Return the loss of regression"""
    losses_tr = []
    losses_te = []

    for f in range(0, k):
        tr_x = []
        tr_y = []
        te_x = []
        te_y = []
        for i in range(0, len(tx)):
            if (i in k_indices[f]):
                te_x.append(tx[i])
                te_y.append(y[i])
            else:
                tr_x.append(tx[i])
                tr_y.append(y[i])

        #tr_poly = build_poly(tr_x, degree)
        #te_poly = build_poly(te_x, degree)

        #w_s, tr_e = ridge_regression(tr_y, tr_poly, lambda_)
        #te_e = compute_mse(te_y, te_poly, w_s)

        tr_x = np.array(tr_x)
        tr_y = np.array(tr_y)
        te_x = np.array(te_x)
        te_y = np.array(te_y)

        w_s, tr_e = ridge_regression(tr_y, tr_x, lambda_)
        te_e = compute_mse(te_y, te_x, w_s)

        #initial_w = np.zeros(shape=(tx.shape[1], 1))
        #max_iters = 100
        #gamma = 0.1
        #threshold = 1e-8

        #(y, tx, lambda_, initial_w, max_iters, gamma, threshold)
        #te_e = calculate_reg_log_loss(te_y, te_x, w_s, lambda_)


        losses_tr.append(np.sqrt(2 * tr_e))
        losses_te.append(np.sqrt(2 * te_e))

        #print('Performed cross validation {}'.format(f))

    loss_tr = sum(losses_tr) / k
    loss_te = sum(losses_te) / k

    return loss_tr, loss_te



def cross_validation_demo(y, tx, seed, k_fold, degree, lambdas):
    #seed = 1
    #degree = 7
    #k_fold = 4
    #lambdas = np.logspace(-4, 0, 30)

    k_indices = build_k_indices(y, k_fold, seed)

    rmse_tr = []
    rmse_te = []

    for i in range(0, len(lambdas)):
        loss_tr, loss_te = cross_validation(y, tx, k_indices, k_fold, lambdas[i], degree)
        rmse_tr.append(loss_tr)
        rmse_te.append(loss_te)

        print('Lambda value {i} (={v}), rmse_tr = {tr}, rmse_te = {te}'.format(i=i, v=lambdas[i], tr=loss_tr, te=loss_te))

        if (i >= 1):
            plt.clf()
            plt.semilogx(lambdas[0:i + 1], rmse_tr, marker=".", color='b', label='train error')
            plt.semilogx(lambdas[0:i + 1], rmse_te, marker=".", color='r', label='test error')
            plt.xlabel("lambda")
            plt.ylabel("rmse")
            #plt.xlim(1e-4, 1)
            plt.title("cross validation")
            plt.legend(loc=2)
            plt.grid(True)
            plt.savefig("cross_validation")
            plt.show()
