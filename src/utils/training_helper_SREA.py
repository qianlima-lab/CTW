import collections
import logging
import os
import shutil
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import cluster
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax

from src.env_test.knn_utils import mine_nearest_neighbors, plt_knn_acc
from src.models.MultiTaskClassification import AEandClass, NonLinClassifier
from src.models.model import CNNAE
from src.utils.log_utils import StreamToLogger
from src.utils.plotting_utils import plot_loss, plot_embedding, visualize_training_loss, plot_results
from src.utils.utils import cluster_accuracy, evaluate_class_recons, reset_seed_, reset_model, SaverSlave, flip_label, \
    append_results_dict, map_losstype, map_abg, remove_empty_dirs
from src.utils.utils import readable

# import tqdm

columns = shutil.get_terminal_size().columns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def linear_comb(w, x1, x2):
    return (1 - w) * x1 + w * x2


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class CentroidLoss(nn.Module):
    """
    Centroid loss - Constraint Clustering loss of SREA
    """

    def __init__(self, feat_dim, num_classes, reduction='mean'):
        super(CentroidLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        self.reduction = reduction
        self.rho = 1.0

    def forward(self, h, y):
        C = self.centers
        norm_squared = torch.sum((h.unsqueeze(1) - C) ** 2, 2)
        # Attractive
        distance = norm_squared.gather(1, y.unsqueeze(1)).squeeze(-1)
        # Repulsive
        logsum = torch.logsumexp(-torch.sqrt(norm_squared), dim=1)
        loss = reduce_loss(distance + logsum, reduction=self.reduction)
        # Regularization
        reg = self.regularization(reduction='sum')
        return loss + self.rho * reg

    def regularization(self, reduction='sum'):
        C = self.centers
        pairwise_dist = torch.cdist(C, C, p=2) ** 2
        pairwise_dist = pairwise_dist.masked_fill(
            torch.zeros((C.size(0), C.size(0))).fill_diagonal_(1).bool().to(device), float('inf'))
        distance_reg = reduce_loss(-(torch.min(torch.log(pairwise_dist), dim=-1)[0]), reduction=reduction)
        return distance_reg


def temperature(x, th_low, th_high, low_val, high_val):
    if x < th_low:
        return low_val
    elif th_low <= x < th_high:
        return (x - th_low) / (th_high - th_low) * (high_val - low_val) + low_val
    else:  # x == th_high
        return high_val


def create_hard_labels(embedding, centers, y_obs, yhat_hist, w_yhat, w_c, w_obs, classes):
    # TODO: add label temporal dynamics

    # yhat from previous metwork prediction. - Network Ensemble
    steps = yhat_hist.size(-1)
    decay = torch.arange(0, steps, 1).float().to(device)
    decay = torch.exp(-decay / 2)
    yhat_hist = yhat_hist * decay
    yhat = yhat_hist.mean(dim=-1) * w_yhat

    # Label from clustering
    distance_centers = torch.cdist(embedding, centers)
    yc = F.softmin(distance_centers, dim=1).detach() * w_c

    # Observed - given - label (noisy)
    yobs = F.one_hot(y_obs, num_classes=classes).float() * w_obs

    # Label combining
    ystar = (yhat + yc + yobs) / 3
    ystar = torch.argmax(ystar, dim=1)
    return ystar


def train_model(model, train_data, epochs, correct, args, saver=None, plot_loss_flag=True, test_data=None,
                dataset_name=None, pic_n=None):
    # Init variables
    network = model.get_name()
    milestone = args.M
    alpha, beta, gamma = args.abg
    rho = args.class_reg
    epsilon = args.entropy_reg
    history_track = args.track
    correct_start = args.correct_start
    correct_end = args.correct_end
    init_centers = args.init_centers
    classes = args.nbins

    avg_train_loss = []
    avg_train_acc = []

    avg_test_acc = []

    # Init losses
    loss_class = nn.CrossEntropyLoss(reduction='none')
    loss_ae = nn.MSELoss(reduction='mean')
    loss_centroids = CentroidLoss(args.embedding_size, classes, reduction='none').to(device)

    optimizer = torch.optim.Adam(
        list(filter(lambda p: p.requires_grad, model.parameters())) + list(loss_centroids.parameters()),
        lr=args.learning_rate, weight_decay=args.l2penalty, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=0.5)

    p = torch.ones(classes).to(device) / classes
    kmeans = cluster.KMeans(n_clusters=classes, random_state=args.seed)
    yhat_hist = torch.zeros(train_data.dataset.tensors[0].size(0), classes, history_track).to(device)

    print('-' * shutil.get_terminal_size().columns)
    s = 'TRAINING MODEL {} WITH {} LOSS - Correction: {}'.format(network, loss_class._get_name(), str(correct))
    print(f'{s}')
    print('-' * shutil.get_terminal_size().columns)

    # Train loop
    # Force exit with Ctrl + C (Keyboard interrupt command)
    try:
        all_losses = []
        all_indices = []

        all_knn_high_acc = []
        all_knn_low_acc = []
        test_f1s = []

        for idx_epoch in range(1, epochs + 1):
            epochstart = time()
            train_loss = []
            train_acc = []
            train_acc_corrected = []
            epoch_losses = torch.Tensor()
            epoch_indices = torch.Tensor()

            # KMeans after the first milestone - Training WarmUp
            if idx_epoch == init_centers:
                # Init cluster centers with KMeans
                embedding = []
                targets = []
                with torch.no_grad():
                    model.eval()
                    loss_centroids.eval()
                    for data, target, _,_ in train_data:
                        data = data.to(device)
                        output = model.encoder(data)
                        embedding.append(output.squeeze(-1).cpu().numpy())
                        targets.append(target.numpy())
                embedding = np.concatenate(embedding, axis=0)

                targets = np.concatenate(targets, axis=0)
                predicted = kmeans.fit_predict(embedding)
                reassignment, accuracy = cluster_accuracy(targets, predicted)
                # predicted_ordered = np.array(list(map(lambda x: reassignment[x], predicted)))
                # Center reordering. Swap keys and values and sort by keys.
                cluster_centers = kmeans.cluster_centers_[
                    list(dict(sorted({y: x for x, y in reassignment.items()}.items())).values())]
                cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True).to(device)
                with torch.no_grad():
                    # initialise the cluster centers
                    loss_centroids.state_dict()["centers"].copy_(cluster_centers)

            # Train
            model.train()
            loss_centroids.train()

            all_data = []
            all_label = []
            all_raw_data = []


            if idx_epoch == 1:
                teacher_idx = None

            for data, target, data_idx,_ in train_data:

                all_raw_data.append(data.reshape(data.shape[0], -1).numpy())
                target = target.to(device)
                data = data.to(device)
                batch_size = data.size(0)

                # Forward
                optimizer.zero_grad()
                out_AE, out_class, embedding = model(data)
                embedding = embedding.squeeze(-1)

                all_data.append(embedding.data.cpu().numpy())
                all_label.append(target.data.cpu().numpy())

                # Accuracy on noisy labels
                prob = F.softmax(out_class, dim=1)
                prob_avg = torch.mean(prob, dim=0)
                train_acc.append((torch.argmax(prob, dim=1) == target).sum().item() / batch_size)
                loss_noisy_labels = loss_class(out_class, target).detach()

                # Track predictions
                alpha_, beta_, gamma_, epsilon_, rho_ = alpha, beta, gamma, epsilon, rho
                w_yhat, w_c, w_obs = 0, 0, 0

                # Correct labels
                if correct:
                    w_yhat = temperature(idx_epoch, th_low=correct_start, th_high=correct_end, low_val=0,
                                         high_val=1 * beta)  # Pred
                    w_c = temperature(idx_epoch, th_low=correct_start, th_high=correct_end, low_val=0,
                                      high_val=1 * gamma)  # Centers
                    w_obs = temperature(idx_epoch, th_low=correct_start, th_high=correct_end, low_val=1,
                                        high_val=0)  # Observed

                    beta_ = temperature(idx_epoch, th_low=init_centers - args.track, th_high=correct_start,
                                        low_val=0, high_val=beta)  # Class
                    gamma_ = temperature(idx_epoch, th_low=init_centers, th_high=correct_start, low_val=0,
                                         high_val=gamma)  # Centers
                    rho_ = temperature(idx_epoch, th_low=init_centers - args.track, th_high=correct_start,
                                       low_val=0, high_val=rho * beta_)  # Lp
                    epsilon_ = temperature(idx_epoch, th_low=init_centers - args.track, th_high=correct_start,
                                           low_val=0, high_val=epsilon * beta_)  # Le

                    ystar = create_hard_labels(embedding, loss_centroids.centers, target, yhat_hist[data_idx],
                                               w_yhat, w_c, w_obs, classes)
                    target = ystar
                else:
                    gamma_ = temperature(idx_epoch, th_low=init_centers, th_high=init_centers, low_val=0,  # Centers
                                         high_val=gamma)
                    rho_ *= beta
                    epsilon_ *= beta

                    gamma_ = 0
                    rho_ = 0
                    epsilon_ = 0

                loss_cntrs_ = loss_centroids(embedding, target)
                loss_class_ = loss_class(out_class, target)
                loss_recons_ = loss_ae(out_AE, data)

                L_p = -torch.sum(torch.log(prob_avg) * p)  # Distribution regularization
                L_e = -torch.mean(torch.sum(prob * F.log_softmax(out_class, dim=1), dim=1))  # Entropy regularization

                loss = alpha_ * loss_recons_ + beta_ * loss_class_.mean() + gamma_ * loss_cntrs_.mean() + \
                       L_p * rho_ + L_e * epsilon_

                # Track losses each sample
                epoch_losses = torch.cat((epoch_losses, loss_noisy_labels.data.detach().cpu()))
                epoch_indices = torch.cat((epoch_indices, data_idx.cpu().float()))
                loss.backward()

                # Append predictions
                yhat_hist[data_idx] = yhat_hist[data_idx].roll(1, dims=-1)
                yhat_hist[data_idx, :, 0] = prob.detach()

                optimizer.step()

                # Update loss  monitor
                train_loss.append(loss.data.item())
                train_acc_corrected.append((torch.argmax(prob, dim=1) == target).sum().item() / batch_size)

            all_data = np.concatenate(all_data)
            all_label = np.concatenate(all_label)
            all_raw_data = np.concatenate(all_raw_data)


            indices, accuracy = mine_nearest_neighbors(features=all_data.astype(np.float32), real_label=all_label, topk=20,
                                                       calculate_accuracy=True)
            if idx_epoch == 1:
                indices, l_accuracy = mine_nearest_neighbors(features=all_raw_data.astype(np.float32), real_label=all_label,
                                                           topk=20,
                                                           calculate_accuracy=True)
                all_knn_low_acc.append(round(l_accuracy, 4))
            all_knn_low_acc.append(round(accuracy, 4))


            scheduler.step()

            test_loss, _ = eval_model(model, test_data, [loss_ae, loss_class, loss_centroids],
                                               [alpha_, beta_, gamma_])
            test_acc, f1 = test_step(test_data, model)

            # calculate average loss over an epoch
            train_loss_epoch = np.average(train_loss)
            avg_train_loss.append(train_loss_epoch)

            train_acc_epoch = 100 * np.average(train_acc)
            train_acc_corr_epoch = 100 * np.average(train_acc_corrected)

            avg_train_acc.append(train_acc_epoch)
            avg_test_acc.append(test_acc)
            test_f1s.append(f1)

            print(
                'Epoch [{}/{}], Time:{:.3f} - TrAcc:{:.3f} - TrAccCorr:{:.3f} - TestAcc:{:.3f} - TrLoss:{:.5f} - '
                'lr:{:.5f} - alpha:{:.3f} - beta:{:.3f} - gamma:{:.3f} - rho:{:.3f} - eps:{:.3f}'
                ' - w_obs:{:.3f} - w_yhat:{:.3f} - w_cen:{:.3f}'
                    .format(idx_epoch, epochs, time() - epochstart, train_acc_epoch, train_acc_corr_epoch,
                            test_acc, train_loss_epoch, optimizer.param_groups[0]['lr'],
                            alpha_, beta_, gamma_, rho_, epsilon_, w_obs, w_yhat, w_c))

            all_losses.append(epoch_losses)
            all_indices.append(epoch_indices)

        # if pic_n is None:
        #     plt_knn_acc(high_acc=all_knn_low_acc, low_acc=all_knn_low_acc, dataset_name=dataset_name, pic_n="only")
        # else:
        #     plt_knn_acc(high_acc=all_knn_low_acc, low_acc=all_knn_low_acc, dataset_name=dataset_name, pic_n=pic_n)

    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    ################################################
    all_losses = np.vstack(all_losses)
    all_indices = np.vstack(all_indices)
    ################################################

    if plot_loss_flag:
        plot_loss(avg_train_loss, loss_class._get_name(), network, kind='loss', saver=saver)
        plot_loss(avg_train_acc,  loss_class._get_name(), network, kind='accuracy', saver=saver)

    if args.plt_loss_hist:
        plot_train_loss_and_test_acc(avg_train_loss,np.array(avg_test_acc)/100,args,pred_precision=np.array(avg_train_acc)/100,
                                     saver=saver,save=True)


    test_results_last_ten_epochs = dict()
    test_results_last_ten_epochs['last_ten_test_acc'] = avg_test_acc[-10:]
    test_results_last_ten_epochs['last_ten_test_f1'] = test_f1s[-10:]

    return model, loss_centroids, (all_losses, all_indices), test_results_last_ten_epochs

def test_step(data_loader, model):
    model = model.eval()

    yhat = []
    ytrue = []

    for x, y in data_loader:
        x = x.to(device)

        _, logits, _ = model(x)

        yhat.append(logits.detach().cpu().numpy())
        try:
            y = y.cpu().numpy()
        except:
            y = y.numpy()
        ytrue.append(y)

    yhat = np.concatenate(yhat,axis=0)
    ytrue = np.concatenate(ytrue,axis=0)
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)
    accuracy = accuracy_score(ytrue, y_hat_labels)
    f1_weighted = f1_score(ytrue, y_hat_labels, average='weighted')

    return accuracy, f1_weighted


def eval_model(model, loader, list_loss, coeffs):
    loss_ae, loss_class, loss_centroids = list_loss
    alpha, beta, gamma = coeffs
    losses = []
    accs = []

    with torch.no_grad():
        model.eval()
        loss_centroids.eval()
        for data in loader:
            # get the inputs
            inputs, target = data  # input shape must be (BS, C, L)
            inputs = Variable(inputs.float()).to(device)
            target = Variable(target.long()).to(device)
            batch_size = inputs.size(0)

            out_AE, out_class, embedding = model(inputs)
            ypred = torch.max(F.softmax(out_class, dim=1), dim=1)[1]

            loss_recons_ = loss_ae(out_AE, inputs)
            loss_class_ = loss_class(out_class, target)
            loss_cntrs_ = loss_centroids(embedding.squeeze(-1), target)
            loss = alpha * loss_recons_ + beta * loss_class_.mean() + gamma * loss_cntrs_.mean()

            losses.append(loss.data.item())

            accs.append((ypred == target).sum().item() / batch_size)

    return np.array(losses).mean(), 100 * np.average(accs)


def train_eval_model(model, x_train, x_test, Y_train, Y_test, Y_train_clean,
                     mask_train, ni, args, saver, correct_labels, plt_embedding=True, plt_loss_hist=True,
                     plt_recons=False, plt_cm=True, dataset_name=None, pic_n=None):
    classes = len(np.unique(Y_train_clean))

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))), torch.from_numpy(mask_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers, pin_memory=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                   num_workers=args.num_workers, pin_memory=True)

    ######################################################################################################
    # Train model
    model, clusterer, (train_losses, train_idxs), test_results_last_ten_epochs = train_model(model, train_loader,
                                                               epochs=args.epochs, args=args, correct=correct_labels,
                                                               saver=saver, plot_loss_flag=args.plt_loss,
                                                            test_data=test_loader,dataset_name=dataset_name, pic_n=pic_n)
    cluster_centers = clusterer.centers.detach().cpu().numpy()
    print('Train ended')
    # print("test_results_last_ten_epochs = ", test_results_last_ten_epochs)

    if plt_embedding:
        plot_embedding(model.encoder, train_eval_loader, cluster_centers, Y_train_clean,
                       Y_train, network='CNN', saver=saver, correct=correct_labels)

    # if ni > 0 and plt_loss_hist:
    #     visualize_training_loss(train_losses, train_idxs, mask_train, 'CNN', classes, ni, saver,
    #                             correct=correct_labels)

    ########################################## Eval ############################################

    # save test_results: test_acc(the last model), test_f1(the last model), avg_last_ten_test_acc, avg_last_ten_test_f1
    test_results = evaluate_class_recons(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
                                         'Test', correct_labels, plt_cm=plt_cm, plt_lables=False, plt_recons=plt_recons)
    test_results['avg_last_ten_test_acc'] = np.mean(test_results_last_ten_epochs['last_ten_test_acc'])
    test_results['avg_last_ten_test_f1'] = np.mean(test_results_last_ten_epochs['last_ten_test_f1'])

    ###############################################################################################

    plt.close('all')
    torch.cuda.empty_cache()
    return test_results


def main_wrapper(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=None,
                 dataset_name=None, pic_n=None):
    classes = len(np.unique(Y_train_clean))
    args.nbins = classes

    # Network definition
    classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                  norm=args.normalization)

    model_ae = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                     seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                     padding=args.padding, dropout=args.dropout, normalization=args.normalization)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = AEandClass(ae=model_ae, classifier=classifier, name='CNN').to(device)

    # print("model = ", model)

    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % ('CNN', readable(nParams))
    print(f'{s}')

    ######################################################################################################
    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train_clean.shape, [(Y_train_clean == i).sum() for i in np.unique(Y_train_clean)])

    print('Test:', x_test.shape, Y_test_clean.shape, [(Y_test_clean == i).sum() for i in np.unique(Y_test_clean)])

    ######################################################################################################
    # Main loop
    df_results = pd.DataFrame()
    if seed is None:
        seed = np.random.choice(1000,1, replace=False)

    args.correct_start = args.init_centers + args.delta_start
    args.correct_end = args.init_centers + args.delta_start + args.delta_end

    print()
    print('#' * shutil.get_terminal_size().columns)
    print('RANDOM SEED:{}'.format(seed).center(columns))
    print('#' * shutil.get_terminal_size().columns)
    print()

    args.seed = seed

    # torch.save(model.state_dict(), os.path.join(saver.path, 'initial_weight.pt'))

    test_results_main = collections.defaultdict(list)
    test_corrected_results_main = collections.defaultdict(list)
    saver_loop = SaverSlave(os.path.join(saver.path, f'seed_{seed}'))
    # saver_loop.append_str(['SEED: {}'.format(seed), '\r\n'])

    ni = args.ni

    saver_slave = SaverSlave(os.path.join(saver.path, f'seed_{seed}', f'ratio_{ni}'))

    correct_labels = args.correct
    # True or false
    print('+' * shutil.get_terminal_size().columns)
    print('Label noise ratio: %.3f' % ni)
    print('Correct labels:', correct_labels)
    print('+' * shutil.get_terminal_size().columns)

    reset_seed_(seed)
    model = reset_model(model)

    Y_train, mask_train = flip_label(x_train, Y_train_clean, ni, args)
    Y_test = Y_test_clean

    # Re-load initial weights
    # model.load_state_dict(torch.load(os.path.join(saver.path, 'initial_weight.pt')))

    test_results = train_eval_model(model, x_train, x_test, Y_train,
                                    Y_test, Y_train_clean,
                                    mask_train, ni, args, saver_slave,
                                    correct_labels,
                                    plt_embedding=args.plt_embedding,
                                    plt_loss_hist=args.plt_loss_hist,
                                    plt_recons=args.plt_recons,
                                    plt_cm=args.plt_cm,
                                    dataset_name=dataset_name,
                                    pic_n=pic_n)
    if correct_labels:
        test_corrected_results_main = append_results_dict(test_corrected_results_main, test_results)
    else:
        test_results_main = append_results_dict(test_results_main, test_results)

    test_results['noise'] = ni
    test_results['noise_type'] = map_losstype(args.label_noise)
    test_results['seed'] = seed
    test_results['correct'] = str(correct_labels)
    test_results['losses'] = map_abg(args.abg)
    test_results['track'] = args.track
    test_results['init_centers'] = args.init_centers
    test_results['delta_start'] = args.delta_start
    test_results['delta_end'] = args.delta_end

    # saver_seed.append_str(['Test Results:'])
    # saver_seed.append_dict(test_results)
    df_results = df_results.append(test_results, ignore_index=True)
    if len(test_results_main):
        keys = list(test_results_main.keys())
    else:
        keys = list(test_corrected_results_main.keys())
    if args.plt_cm:
        fig_title = f"Data:{args.dataset} - Loss:{map_abg(args.abg)} - classes:{classes} - noise:{map_losstype(args.label_noise)}"
        plot_results(df_results.loc[df_results.seed == seed], keys, saver_loop, title=fig_title,
                     x='noise', hue='correct', col=None, kind='bar', style='whitegrid')
    if args.plt_cm:
        fig_title = f"Dataset:{args.dataset} - Loss:{map_abg(args.abg)} - classes:{classes} - noise:{map_losstype(args.label_noise)}"
        plot_results(df_results, keys, saver, title=fig_title,
                     x='noise', hue='correct', col=None, kind='box', style='whitegrid')

    remove_empty_dirs(saver.path)

    return test_results



def get_score(singular_vector_dict, features, labels, normalization=True):
    '''
    Calculate the score providing the degree of showing whether the data is clean or not.
    '''
    if normalization:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat / np.linalg.norm(feat))) for indx, feat in
                  enumerate(tqdm(features))]
    else:
        scores = [np.abs(np.inner(singular_vector_dict[labels[indx]], feat)) for indx, feat in
                  enumerate(tqdm(features))]

    return np.array(scores)


def get_features(model, dataloader,teacher_idx = None):
    '''
    Concatenate the hidden features and corresponding labels
    '''
    labels = np.empty((0,))
    data_idxs = np.empty((0,))

    model.eval()
    model.cuda()
    with tqdm(dataloader) as progress:
        for batch_idx, (data, label, index) in enumerate(progress):

            label = label.long()

            if teacher_idx is not None:
                index = list(index.numpy())
                data = data.detach().cpu().numpy()
                label = list(label.numpy())

                for i, ix in enumerate(index):
                    if ix not in teacher_idx:
                        index.remove(ix)
                        data = np.delete(data, i, axis=0)
                        label.pop(i)

                data = torch.tensor(data)
                label = torch.tensor(label)
                # index = torch.tensor(index)

            data = data.cuda()
            feature = model.encoder(data).squeeze(-1)

            labels = np.concatenate((labels, label.cpu()))
            data_idxs = np.concatenate((data_idxs,index))
            if batch_idx == 0:
                features = feature.detach().cpu()

            else:
                features = np.concatenate((features, feature.detach().cpu()), axis=0)

    return features, labels, data_idxs


def cleansing(scores, labels):
    '''
    Assume the distribution of scores: bimodal spherical distribution.

    return clean labels
    that belongs to the clean cluster made by the KMeans algorithm
    '''

    indexes = np.array(range(len(scores)))
    clean_labels = []
    for cls in np.unique(labels):
        cls_index = indexes[labels == cls]
        kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(scores[cls_index].reshape(-1, 1))
        if np.mean(scores[cls_index][kmeans.labels_ == 0]) < np.mean(
            scores[cls_index][kmeans.labels_ == 1]): kmeans.labels_ = 1 - kmeans.labels_

        clean_labels += cls_index[kmeans.labels_ == 0].tolist()

    return np.array(clean_labels, dtype=np.int64)


def get_singular_vector(features, labels):
    '''
    To get top1 sigular vector in class-wise manner by using SVD of hidden feature vectors
    features: hidden feature vectors of data (numpy)
    labels: correspoding label list
    '''

    singular_vector_dict = {}
    with tqdm(total=len(np.unique(labels))) as pbar:
        for index in np.unique(labels):
            _, _, v = np.linalg.svd(features[labels == index])
            singular_vector_dict[index] = v[0]
            pbar.update(1)

    return singular_vector_dict

def plot_train_loss_and_test_acc(avg_train_losses,test_acc_list,args,pred_precision=None,ori_pred_precision=None,saver=None,save=False):
    fig = plt.figure(figsize=(1200,800))

    plt.gcf().set_facecolor(np.ones(3) * 240 / 255)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    l1 = ax1.plot(avg_train_losses,'-', c='orangered', label='Training loss', linewidth=1)
    l2 = ax2.plot(test_acc_list, '-', c='blue', label='Test acc', linewidth=1)
    l3 = ax2.plot(pred_precision,'-',c='green',label='train Pred acc',linewidth=1)

    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs,loc='upper right')
    # plt.legend(handles=[l1,l2],labels=["Training loss","Test acc"],loc='upper right')

    plt.axvline(args.warmup,color='g',linestyle='--')

    ax1.set_xlabel('epoch',  size=18)
    ax1.set_ylabel('Train loss',size=18)
    ax2.set_ylabel('Test acc',  size=18)
    plt.gcf().autofmt_xdate()
    plt.title(f'Model:new model dataset:{args.dataset}')
    plt.grid(True)

    plt.tight_layout()

    saver.save_fig(fig, name=args.dataset)

