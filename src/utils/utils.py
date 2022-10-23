import argparse
import argparse
import os
import shutil
from math import exp
from math import inf

import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    classification_report
from sklearn.mixture import GaussianMixture as GMM
from torch.autograd import Variable

# from src.utils.plotting_utils import plot_cm
# import src.utils.plotting_utils as plt
from src.utils.metrics import evaluate_multi
# import src.utils.plotting_utils as plt
from src.utils.plotting_utils import *
from src.utils.saver import Saver

######################################################################################################################
columns = shutil.get_terminal_size().columns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import matplotlib.pyplot as plt

def plot_cm(cm, T=None, network='Net', title_str='', saver=None):
    classes = cm.shape[0]
    acc = np.diag(cm).sum() / cm.sum()
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if T is not None:
        fig, (ax, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4 + classes / 2.5, 4 + classes / 1.25))
        T_norm = T.astype('float') / T.sum(axis=1)[:, np.newaxis]
        # Transition matrix ax
        sns.heatmap(T_norm, annot=T_norm, cmap=plt.cm.YlGnBu, cbar=False, ax=ax2, linecolor='black', linewidths=0)
        ax2.set(ylabel='Noise Transition Matrix')
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4 + classes / 2.5, 4 + classes / 2.5))

    # Cm Ax
    sns.heatmap(cm_norm, annot=None, cmap=plt.cm.YlGnBu, cbar=False, ax=ax, linecolor='black', linewidths=0)
    # ax.imshow(cm_norm, aspect='auto', interpolation='nearest', cmap=plt.cm.YlGnBu)
    # ax.matshow(cm_norm, cmap=plt.cm.Blues)

    ax.set(title=f'Model:{network} - Accuracy:{100 * acc:.1f}% - {title_str}',
           ylabel='Confusion Matrix (Predicted / True)',
           xlabel=None)
    # ax.set_ylim([1.5, -0.5])
    thresh = cm_norm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, '%d (%.2f)' % (cm[i, j], cm_norm[i, j]),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black")

    fig.tight_layout()

    if saver:
        saver.save_fig(fig, f'CM_{title_str}')


from src.utils.decorators import repeat

# def reset_rng(func):
#     """
#     decorator wrapper to reset the numpy rng
#     """
#
#     @functools.wraps(func)
#     def wrapper_decorator(*args, **kwargs):
#         np.random.seed(SEED)
#         print(f'Reset RNG. Seed:{SEED}.')
#         value = func(*args, **kwargs)
#         # Do something after
#         return value
#
#     return wrapper_decorator


# @reset_rng
@repeat(num_times=1)
def plot_prediction(X, X_hat, nrows=5, ncols=5, figsize=(19.2, 10), title: str = 'model', saver: object = None,
                    figname: str = ''):
    # TODO: export indices to havve a fair comparison across different methods
    # Setting seed for reproducibility.
    idx = np.random.randint(0, X_hat.shape[0], nrows * ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight=600)

    for i, ax in enumerate(axes.ravel()):
        ax.plot(X[idx[i]], '--', label='Original')
        ax.plot(X_hat[idx[i]], label='Recons')
        # ax.set_ylim([-0.05, 1.05])
        # if i == ncols//2:
        #    ax.set_title(title, fontsize=12, fontweight=600)

    plt.legend()
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    # plt.show()

    if saver:
        saver.save_fig(fig, figname + '_pred')

#####################################################################################################################

class SaverSlave(Saver):
    def __init__(self, path):
        super(Saver)

        self.path = path
        self.makedir_()
        # self.make_log()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_ziplen(l, n):
    if len(l) % n != 0:
        l += [l[-1]]
        return check_ziplen(l, n)
    else:
        return l


def remove_duplicates(sequence):
    unique = []
    [unique.append(item) for item in sequence if item not in unique]
    return unique


def map_abg(x):
    if x == [0, 1, 0]:
        return r'$\mathcal{L}_c$'
    elif x == [1, 0, 0]:
        return r'$\mathcal{L}_{ae}$'
    elif x == [1, 1, 0]:
        return r'$\mathcal{L}_c + \mathcal{L}_{ae}$'
    elif x == [0, 1, 1]:
        return r'$\mathcal{L}_c + \mathcal{L}_{cc}$'
    elif x == [1, 1, 1]:
        return r'$\mathcal{L}_c + \mathcal{L}_{ae} + \mathcal{L}_{cc}$'
    else:
        raise ValueError


def map_losstype(x):
    if x == 0:
        return 'Symm'
    else:
        return 'Asymm_{}'.format(x)


def map_abg_main(x):
    if x is None:
        return 'Variable'
    else:
        return '_'.join([str(int(j)) for j in x])


def remove_empty_dir(path):
    try:
        os.rmdir(path)
    except OSError:
        pass


def remove_empty_dirs(path):
    for root, dirnames, filenames in os.walk(path, topdown=False):
        for dirname in dirnames:
            remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))


def add_noise(x, sigma=0.2, mu=0.):
    noise = mu + torch.randn(x.size()) * sigma
    noisy_x = x + noise
    return noisy_x


def readable(num):
    for unit in ['', 'k', 'M']:
        if abs(num) < 1e3:
            return "%3.3f%s" % (num, unit)
        num /= 1e3
    return "%.1f%s" % (num, 'B')


# Unique labels
def categorizer(y_cont, y_discrete):
    Yd = np.diff(y_cont, axis=0)
    Yd = (Yd > 0).astype(int).squeeze()
    C = pd.Series([x + y for x, y in
                   zip(list(y_discrete[1:].astype(int).astype(str)), list((Yd).astype(str)))]).astype(
        'category')
    return C.cat.codes


def reset_seed_(seed):
    # Resetting SEED to fair comparison of results
    print('Settint seed: {}'.format(seed))
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def reset_model(model):
    print('Resetting model parameters...')
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model


def append_results_dict(main, sub):
    for k, v in zip(sub.keys(), sub.values()):
        main[k].append(v)
    return main


def flip_label(train_data,target, ratio, args):
    """
    Induce label noise by randomly corrupting labels
    :param target: list or array of labels
    :param ratio: float: noise ratio
    :param pattern: flag to choose which type of noise.
            0 or mod(pattern, #classes) == 0 = symmetric
            int = asymmetric
            -1 = inst
            -2 = flip
    :return:
    """
    pattern = args.label_noise
    np.random.seed(123)
    assert 0 <= ratio < 1

    target = np.array(target).astype(int)
    label = target.copy()
    n_class = args.nbins

    if abs(ratio-0)<1e-5:
        return label,np.array([int(x != y) for (x, y) in zip(target, label)])

    if pattern==-1:
        # label,mask=noisify_instance(train_data, target, ratio,n_class)
        label,mask=get_instance_noisy_label(ratio, train_data,target,
                                 n_class, norm_std=0.1, seed=123)
    elif type(pattern) is int:
        for i in range(label.shape[0]):
            # symmetric noise
            if (pattern % n_class) == 0:
                p1 = ratio / (n_class - 1) * np.ones(n_class)
                p1[label[i]] = 1 - ratio
                label[i] = np.random.choice(n_class, p=p1)
            elif pattern > 0:
                # Asymm
                label[i] = np.random.choice([label[i], (target[i] + pattern) % n_class], p=[1 - ratio, ratio])
            else:
                # Flip noise
                label[i] = np.random.choice([label[i], 0], p=[1 - ratio, ratio])

    elif type(pattern) is str:
        raise ValueError

    mask = np.array([int(x != y) for (x, y) in zip(target, label)])

    return label, mask


def cluster_accuracy(y_true, y_predicted, cluster_number=None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
                max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy


def evaluate_model_multi(model, dataloder, y_true, x_true,
                         metrics=('mae', 'mse', 'rmse', 'std_ae', 'smape', 'rae', 'mbrae', 'corr', 'r2')):
    xhat, yhat = predict_multi(model, dataloder)

    # Classification
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)

    accuracy = accuracy_score(y_true, y_hat_labels)
    f1_weighted = f1_score(y_true, y_hat_labels, average='weighted')

    cm = confusion_matrix(y_true, y_hat_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    report = classification_report(y_true, y_hat_labels)
    print(report)

    # AE
    residual = xhat - x_true
    results = evaluate_multi(actual=x_true, predicted=xhat, metrics=metrics)

    return report, y_hat_proba, y_hat_labels, accuracy, f1_weighted, xhat, residual, results


def evaluate_model(model, dataloder, y_true):
    yhat = predict(model, dataloder)

    # Classification
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)

    accuracy = accuracy_score(y_true, y_hat_labels)
    f1_weighted = f1_score(y_true, y_hat_labels, average='weighted')

    cm = confusion_matrix(y_true, y_hat_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    report = classification_report(y_true, y_hat_labels)
    print(report)

    return report, y_hat_proba, y_hat_labels, accuracy, f1_weighted


def evaluate_class_recons(model, x, Y, Y_clean, dataloader, ni, saver, network='Model', datatype='Train', correct=False,
                          plt_cm=True, plt_lables=True, plt_recons=True):
    print(f'{datatype} score')
    if Y_clean is not None:
        T = confusion_matrix(Y_clean, Y)
    else:
        T = None
    results_dict = dict()

    title_str = f'{datatype} - ratio:{ni} - correct:{str(correct)}'

    results, yhat_proba, yhat, acc, f1, recons, _, ae_results = evaluate_model_multi(model, dataloader, Y, x)

    if plt_cm:
        # plt.plot_cm(confusion_matrix(Y, yhat), T, network=network,
        #             title_str=title_str, saver=saver)
        plot_cm(confusion_matrix(Y, yhat), T, network=network,
                    title_str=title_str, saver=saver)
    if plt_lables:
        # plt.plot_pred_labels(Y, yhat, acc, residuals=None, dataset=f'{datatype}. noise:{ni}', saver=saver)
        plot_pred_labels(Y, yhat, acc, residuals=None, dataset=f'{datatype}. noise:{ni}', saver=saver)
    if plt_recons:
        # plt.plot_prediction(x, recons, nrows=5, ncols=5, figsize=(19.2, 10.80), saver=saver,
        #                     title=f'{datatype} data: mse:%.4f rmse:%.4f corr:%.4f R2:%.4f' % (
        #                         ae_results['mse'], ae_results['rmse'],
        #                         ae_results['corr'], ae_results['r2']), figname=f'AE_{datatype}')
        plot_prediction(x, recons, nrows=5, ncols=5, figsize=(19.2, 10.80), saver=saver,
                            title=f'{datatype} data: mse:%.4f rmse:%.4f corr:%.4f R2:%.4f' % (
                                ae_results['mse'], ae_results['rmse'],
                                ae_results['corr'], ae_results['r2']), figname=f'AE_{datatype}')

    results_dict['acc'] = acc
    results_dict['f1_weighted'] = f1
    # saver.append_str([f'{datatype}Set', 'Classification report:', results])
    # saver.append_str(['AutoEncoder results:'])
    # saver.append_dict(ae_results)
    return results_dict


def evaluate_class(model, x, Y, Y_clean, dataloader, ni, saver, network='Model', datatype='Train', correct=False,
                   plt_cm=True, plt_lables=True):
    print(f'{datatype} score')
    if Y_clean is not None:
        T = confusion_matrix(Y_clean, Y)
    else:
        T = None
    results_dict = dict()

    title_str = f'{datatype} - ratio:{ni} - correct:{str(correct)}'

    results, yhat_proba, yhat, acc, f1 = evaluate_model(model, dataloader, Y)

    if plt_cm:
        # plt.plot_cm(confusion_matrix(Y, yhat), T, network=network,
        #             title_str=title_str, saver=saver)
        plot_cm(confusion_matrix(Y, yhat), T, network=network,
                    title_str=title_str, saver=saver)
    if plt_lables:
        # plt.plot_pred_labels(Y, yhat, acc, residuals=None, dataset=f'{datatype}. noise:{ni}', saver=saver)
        plot_pred_labels(Y, yhat, acc, residuals=None, dataset=f'{datatype}. noise:{ni}', saver=saver)

    results_dict['acc'] = acc
    results_dict['f1_weighted'] = f1
    # saver.append_str([f'{datatype}Set', 'Classification report:', results])
    return results_dict


def predict(model, test_data):
    prediction = []
    with torch.no_grad():
        model.eval()
        for data in test_data:
            data = data[0]
            data = data.float().to(device)
            output = model(data)
            try:
                prediction.append(output.cpu().numpy())
            except:
                prediction.append(output[0].cpu().numpy())

    prediction = np.concatenate(prediction, axis=0)
    return prediction


def predict_multi(model, test_data):
    reconstruction = []
    prediction = []
    with torch.no_grad():
        model.eval()
        for data in test_data:
            data = data[0].float().to(device)
            out_ae, out_class, embedding = model(data)
            prediction.append(out_class.cpu().numpy())
            reconstruction.append(out_ae.cpu().numpy())

    prediction = np.concatenate(prediction, axis=0)
    reconstruction = np.concatenate(reconstruction, axis=0)
    return reconstruction, prediction

def noisify_instance(train_data,train_labels,noise_rate,num_class):

    np.random.seed(0)

    q_ = np.random.normal(loc=noise_rate,scale=0.1,size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q)==50000:
            break
    S=train_data[0].flatten().shape[0]
    w = np.random.normal(loc=0,scale=1,size=(S,num_class))

    noisy_labels = []
    for i, sample in enumerate(train_data):
        sample = sample.flatten()
        p_all = np.matmul(sample,w)
        p_all[train_labels[i]] = -1000000
        p_all = q[i]* F.softmax(torch.tensor(p_all),dim=0).numpy()
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(np.arange(num_class),p=p_all/sum(p_all)))
    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum())/len(train_labels)
    print("##############################\n over all real noise rate: {}\n##############################".format(over_all_noise_rate))
    mask = np.array([int(x != y) for (x, y) in zip(train_labels, noisy_labels)])
    return noisy_labels, mask

def get_instance_noisy_label(n, train_data,labels, num_classes, norm_std, seed):
    # n -> noise_rate
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed
    print("building dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))


    feature_size=train_data[0].flatten().shape[0]
    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    # if isinstance(labels, list):
    labels = torch.FloatTensor(labels)
    labels = labels.cuda()
    train_data=torch.FloatTensor(train_data)
    train_data=train_data.cuda()

    W = np.random.randn(label_num, feature_size, label_num)


    W = torch.FloatTensor(W).cuda()
    for i, (x, y) in enumerate(zip(train_data,labels)):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        y=int(y)
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1


    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break

    mask = np.array([int(x != y) for (x, y) in zip(labels, new_label)])
    return np.array(new_label),mask



def fit_mixture_bmm(scores, labels, p_threshold=0.5):
    """
    Assum the distribution of scores: bimodal beta mixture model

    return clean labels
    that belongs to the clean cluster by fitting the score distribution to BMM
    """

    clean_labels = []
    indexes = np.array(range(len(scores)))
    for cls in np.unique(labels):
        cls_index = indexes[labels == cls]
        feats = scores[labels == cls]
        feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
        feats_ = (feats_ - feats_.min()) / (feats_.max() - feats_.min())
        bmm = BetaMixture(max_iters=100)
        bmm.fit(feats_)

        mean_0 = bmm.alphas[0] / (bmm.alphas[0] + bmm.betas[0])
        mean_1 = bmm.alphas[1] / (bmm.alphas[1] + bmm.betas[1])
        clean = 0 if mean_0 > mean_1 else 1

        init = bmm.predict(feats_.min(), p_threshold, clean)
        for x in np.linspace(feats_.min(), feats_.max(), 50):
            pred = bmm.predict(x, p_threshold, clean)
            if pred != init:
                bound = x
                break

        clean_labels += [cls_index[clean_idx] for clean_idx in range(len(cls_index)) if feats[clean_idx] > bound]

    return np.array(clean_labels, dtype=np.int64)


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar) ** 2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x, threshold, clean):
        return self.posterior(x, clean) > threshold

    def create_lookup(self, y):
        x_l = np.linspace(0 + self.eps_nan, 1 - self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l  # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

def hyperbolic_tangent(y1,y2,epochs,warmup):
    amplitude = y2-y1
    e = exp(1)
    x = np.arange(0,epochs,1)
    y = (e ** (0.05 * (x - 0.5 * epochs)) - e ** (-0.05 * (x - 0.5 * epochs))) / (
                e ** (0.05 * (x - 0.5 * epochs)) + e ** (-0.05 * (x - 0.5 * epochs))) * 0.5 * amplitude + 0.5 * amplitude
    y[:warmup] = [0]*warmup
    return y


def loss_cores(epoch, y, t, class_list, ind, noise_or_not, loss_all, loss_div_all, noise_prior=None):
    beta = f_beta(epoch)
    # if epoch == 1:
    #     print(f'current beta is {beta}')
    loss = F.cross_entropy(y, t, reduce=False)
    loss_numpy = loss.data.cpu().numpy()
    num_batch = len(loss_numpy)
    loss_v = np.zeros(num_batch)
    loss_div_numpy = float(np.array(0))
    loss_ = -torch.log(F.softmax(y) + 1e-8)
    # sel metric
    loss_sel = loss - torch.mean(loss_, 1)
    if noise_prior is None:
        loss = loss - beta * torch.mean(loss_, 1)
    else:
        loss = loss - beta * torch.sum(torch.mul(noise_prior, loss_), 1)

    loss_div_numpy = loss_sel.data.cpu().numpy()
    loss_all[ind, epoch] = loss_numpy
    loss_div_all[ind, epoch] = loss_div_numpy
    for i in range(len(loss_numpy)):
        if epoch <= 25:
            loss_v[i] = 1.0
        elif loss_div_numpy[i] <= 0:
            loss_v[i] = 1.0
    loss_v = loss_v.astype(np.float32)
    loss_v_var = Variable(torch.from_numpy(loss_v)).cuda()
    loss_ = loss_v_var * loss
    if sum(loss_v) == 0.0:
        return torch.mean(loss_) / 100000000
    else:
        return torch.sum(loss_) / sum(loss_v), loss_v.astype(int)


def f_beta(epoch):
    beta1 = np.linspace(0.0, 0.0, num=25)
    beta2 = np.linspace(0.0, 2, num=45)
    beta3 = np.linspace(2, 2, num=80)

    beta = np.concatenate((beta1, beta2, beta3), axis=0)
    return beta[epoch]

def compute_noise_prior(Y_train, args):
    '''compute noisy prior propability'''
    idx_each_noisy_label = [[] for i in range(args.nbins)]
    for i in range(args.num_training_samples):
        idx_each_noisy_label[Y_train[i]].append(i)
    num_each_noisy_label = [len(idx_each_noisy_label[i]) for i in range(args.nbins)]
    noisy_prior = np.array(num_each_noisy_label)/args.num_training_samples
    return noisy_prior

def sigua_loss(model_loss, rt, bad_weight,last_sel_id=None,current_batch_idx=None,args=None):
    '''
    :param model_loss:
    :param rt:
    :param bad_weight:
    :param last_sel:
    :param current_batch_idx: if give current_batch_idx, it will only regard the data which is viewed as good last time
    but bad this time as bad.
    :return:
    '''
    num_total_data = int(model_loss.size(0))
    num_good_data = int(num_total_data * rt)
    num_bad_data = int((num_total_data - num_good_data) / 2)  # TODO: Bad data ratio should be changed

    if args.model in ['sigua'] or last_sel_id is None or current_batch_idx is None:
        _, good_data_idx = torch.topk(model_loss, k=num_good_data, largest=False)  # small loss samples
        _, good_and_bad_data_idx = torch.topk(model_loss, k=num_good_data + num_bad_data,
                                          largest=False)  # small loss samples
        bad_data_idx = good_and_bad_data_idx[num_good_data:]
        model_loss_filter = torch.zeros((model_loss.size(0))).cuda()
        model_loss_filter[good_data_idx] = 1.0  # good data
        model_loss_filter[bad_data_idx] = -1.0 * bad_weight  # bad data

    model_loss = (model_loss_filter * model_loss).mean()

    return model_loss,current_batch_idx[good_data_idx].cpu().numpy()

def create_synthetic_dataset(pattern_len=[0.25], pattern_pos=[0.1, 0.65], ts_len=128, ts_n=128):
    random.seed(1234)
    np.random.seed(1234)

    nb_classes = len(pattern_pos) * len(pattern_len)


    x_train = np.random.normal(0.0, 0.1, size=(ts_n, ts_len))
    # x_test = np.random.normal(0.0, 0.1, size=(ts_n, ts_len))

    y_train = np.random.randint(low=0, high=nb_classes, size=(ts_n,))
    # y_test = np.random.randint(low=0, high=nb_classes, size=(ts_n,))

    # make sure at least each class has one example
    y_train[:nb_classes] = np.arange(start=0, stop=nb_classes, dtype=np.int32)
    # y_test[:nb_classes] = np.arange(start=0, stop=nb_classes, dtype=np.int32)

    # each class is defined with a certain combination of pattern_pos and pattern_len
    # with one pattern_len and two pattern_pos we can create only two classes
    # example:  class 0 _____-_  & class 1 _-_____

    # create the class definitions
    class_def = [None for i in range(nb_classes)]

    idx_class = 0
    for pl in pattern_len:
        for pp in pattern_pos:
            class_def[idx_class] = {'pattern_len': int(pl * ts_len),
                                    'pattern_pos': int(pp * ts_len)}
            idx_class += 1

    # create the dataset
    for i in range(ts_n):
        # for the train
        c = y_train[i]
        curr_pattern_pos = class_def[c]['pattern_pos']
        curr_pattern_len = class_def[c]['pattern_len']
        x_train[i][curr_pattern_pos:curr_pattern_pos + curr_pattern_len] = \
            x_train[i][curr_pattern_pos:curr_pattern_pos + curr_pattern_len] + 1.0

        # for the test
        # c = y_test[i]
        # curr_pattern_pos = class_def[c]['pattern_pos']
        # curr_pattern_len = class_def[c]['pattern_len']
        # x_test[i][curr_pattern_pos:curr_pattern_pos + curr_pattern_len] = \
        #     x_test[i][curr_pattern_pos:curr_pattern_pos + curr_pattern_len] + 1.0

    # znorm
    x_train = (x_train - x_train.mean(axis=1, keepdims=True)) \
              / x_train.std(axis=1, keepdims=True)

    # x_test = (x_test - x_test.mean(axis=1, keepdims=True)) \
    #          / x_test.std(axis=1, keepdims=True)

    # visualize example
    # plt.figure()
    # colors = generate_array_of_colors(nb_classes)
    # for c in range(nb_classes):
    #     plt.plot(x_train[y_train == c][0], color=colors[c], label='class-' + str(c))
    # plt.legend(loc='best')
    # plt.savefig('out.pdf')
    # exit()

    # np.save(out_dir+'x_train.npy',x_train)
    # np.save(out_dir+'y_train.npy',y_train)
    # np.save(out_dir+'x_test.npy',x_test)
    # np.save(out_dir+'y_test.npy',y_test)

    # print('Done creating dataset!')

    return np.expand_dims(x_train,axis=-1), y_train

def to_one_hot(classes,y):
    if y.dtype in ['float64','float','float32']:
        y=y.astype(int)
    return(np.eye(classes)[y])

def small_loss_criterion_auto_rate2(model_loss,loss_all=None,args=None,epoch=None,x_idxs=None,labels=None,y_clean=None,
                                    estimate_noise_rate=None):
    '''
        select confident samples by EPS
    '''
    def standardization(data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data - mu) / sigma
    def standardization_minmax(data):
        mini = np.min(data, axis=0)
        maxi = np.max(data, axis=0)
        return (data - mini) / (maxi-mini)

    gamma = args.gamma

    if args.mean_loss_len > args.warmup:
        loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
            loss_all[x_idxs, (epoch - args.warmup + 1):epoch].mean(axis=1))
    else:
        loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
            loss_all[x_idxs, (epoch - args.mean_loss_len + 1):epoch].mean(axis=1))

    labels_numpy = labels.detach().cpu().numpy()
    recreate_idx=torch.tensor([]).long()
    batch_idxs = torch.tensor(np.arange(len(model_loss))).long()
    standar_loss = np.array([])
    for i in range(args.nbins):
        if (labels_numpy==i).sum()>1:
            if args.standardization_choice == 'z-score':
                each_label_loss = standardization(loss_mean[labels_numpy==i])
            else:
                each_label_loss = standardization_minmax(loss_mean[labels_numpy == i])
            standar_loss = np.concatenate((standar_loss,each_label_loss))
            recreate_idx=torch.cat((recreate_idx,batch_idxs[labels_numpy==i]))
        elif (labels_numpy==i).sum()==1:
            standar_loss = np.concatenate((standar_loss, [0.]))
            recreate_idx=torch.cat((recreate_idx,batch_idxs[labels_numpy==i]))

    _, model_sm_idx = torch.topk(torch.from_numpy(standar_loss), k=int(standar_loss.size*(standar_loss<=standar_loss.mean()).mean()), largest=False)


    model_sm_idxs = recreate_idx[model_sm_idx]

    model_loss_filter = torch.zeros((model_loss.size(0))).to(device)
    model_loss_filter[model_sm_idxs] = 1.0
    model_loss = (model_loss_filter * model_loss).sum()

    return model_loss, model_sm_idxs

def small_loss_criterion_auto_rate1(model_loss,loss_all=None,args=None,epoch=None,x_idxs=None,labels=None,p_threshold=0.5):
    '''
    select confident samples class by class:
        auto_rate == 3 means select samples by GMM,
        otherwise (auto_rate == 1) select confident samples using average loss.
    '''
    gamma = args.gamma
    if args.mean_loss_len > args.warmup:
        loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
            loss_all[x_idxs, (epoch - args.warmup + 1):epoch].mean(axis=1))
    else:
        loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
            loss_all[x_idxs, (epoch - args.mean_loss_len + 1):epoch].mean(axis=1))

    labels_numpy = labels.detach().cpu().numpy()
    all_sm_idx = torch.tensor([]).long()
    batch_idx = torch.tensor(np.arange(len(model_loss))).long()
    indexes = np.array(range(len(loss_mean)))

    rate=(loss_mean<=loss_mean.mean()).mean()

    for i in range(args.nbins):
        if (labels_numpy==i).sum()>1:
            each_label_loss = loss_mean[labels_numpy==i]
            if args.auto_rate==3:
                clean_labels = []

                cls_index = indexes[labels_numpy == i]
                feats = each_label_loss
                feats_ = np.ravel(feats).astype(np.float).reshape(-1, 1)
                gmm = GMM(n_components=2, covariance_type='full', tol=1e-6, max_iter=100)
                gmm.fit(feats_)
                prob = gmm.predict_proba(feats_)
                prob = prob[:, gmm.means_.argmin()]
                clean_labels += [clean_idx for clean_idx in range(len(cls_index)) if
                                 prob[clean_idx] > p_threshold]
                model_sm_idx = torch.tensor(clean_labels).long()
            else:
                _, model_sm_idx = torch.topk(torch.from_numpy(each_label_loss), k=int(each_label_loss.size*rate), largest=False)
            all_sm_idx=torch.concat((all_sm_idx,batch_idx[labels_numpy==i][model_sm_idx]))
        elif (labels_numpy==i).sum()==1:
            all_sm_idx=torch.concat((all_sm_idx,batch_idx[labels_numpy==i]))

    model_loss_filter = torch.zeros((model_loss.size(0))).to(device)
    model_loss_filter[all_sm_idx] = 1.0
    model_loss = (model_loss_filter * model_loss).sum()


    return model_loss, all_sm_idx

def small_loss_criterion_without_eliminate(model_loss, rt,loss_all=None,args=None,epoch=None,x_idxs=None,estimate_noise_rate=None):
    '''
    select confident samples w/o EPS
    '''
    if loss_all is None:
        _, model_sm_idx = torch.topk(model_loss, k=int(int(model_loss.size(0)) * rt), largest=False)
    else:
        gamma = args.gamma
        if args.mean_loss_len>args.warmup:
            loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
                loss_all[x_idxs, (epoch - args.warmup + 1):epoch].mean(axis=1))
        else:
            loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
                loss_all[x_idxs, (epoch - args.mean_loss_len + 1):epoch].mean(axis=1))

    _, model_sm_idx = torch.topk(torch.from_numpy(loss_mean), k=int(model_loss.size(0) * ((model_loss <= model_loss.mean()).cpu().numpy().mean())), largest=False)

    model_loss_filter = torch.zeros((model_loss.size(0))).to(device)
    model_loss_filter[model_sm_idx] = 1.0
    model_loss = (model_loss_filter * model_loss).sum()

    return model_loss, model_sm_idx

def small_loss_criterion(model_loss, rt,loss_all=None,args=None,epoch=None,x_idxs=None):
    '''
    select confident samples by providing noise rate
    '''
    if loss_all is None:
        _, model_sm_idx = torch.topk(model_loss, k=int(int(model_loss.size(0)) * rt), largest=False)
    else:
        gamma = args.gamma
        if args.mean_loss_len>args.warmup:
            loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
                loss_all[x_idxs, (epoch - args.warmup + 1):epoch].mean(axis=1))
        else:
            loss_mean = gamma * loss_all[x_idxs, epoch] + (1 - gamma) * (
                loss_all[x_idxs, (epoch - args.mean_loss_len + 1):epoch].mean(axis=1))

        _, model_sm_idx = torch.topk(torch.from_numpy(loss_mean), k=int(int(model_loss.size(0)) * rt), largest=False)

    model_loss_filter = torch.zeros((model_loss.size(0))).to(device)
    model_loss_filter[model_sm_idx] = 1.0
    model_loss = (model_loss_filter * model_loss).sum()
    return model_loss, model_sm_idx

def plot_loss_density(standar_loss,clean_y,observed_y,args=None,epoch=None):

    fig = plt.figure(figsize=(4*args.nbins,3))
    for i in range(args.nbins):
        fig.add_subplot(1, args.nbins, i+1)
        sns.set_style("white")
        sns.distplot(standar_loss[(observed_y==clean_y)&(observed_y==i)], hist=False,
                 kde_kws={'color': 'g', 'linestyle': '-', 'shade': True},
                 norm_hist=True, label='Clean')
        sns.distplot(standar_loss[(observed_y!=clean_y)&(observed_y==i)], hist=False,
                 kde_kws={'color': 'b', 'linestyle': '-', 'shade': True},
                 norm_hist=True, label='Noisy')
    # plt.rcParams.update({'font.size':16})
        plt.legend(fontsize=10)
        plt.tick_params(width=0.5, labelsize=8)
        plt.xlabel('Loss', fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.title('Label {}'.format(i))
    fig.tight_layout()
    fig.suptitle('Epoch {}'.format(epoch))
    plt.savefig(os.path.join(args.basicpath,'src','visualization','{}_epoch{}_{}.png'.format(args.dataset,epoch,
                                                                                             args.standardization_choice)))
    # plt.show()