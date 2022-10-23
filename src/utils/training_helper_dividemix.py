from __future__ import print_function

import collections
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import sys
import random
import shutil
from itertools import chain
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.mixture import GaussianMixture
import tsaug

from src.models.MultiTaskClassification import MetaModel, NonLinClassifier
from src.models.model import CNNAE
from src.utils.plotting_utils import plot_results, plot_embedding
from src.utils.saver import Saver
from src.utils.utils import readable, reset_seed_, reset_model, flip_label, map_abg, remove_empty_dirs, \
    evaluate_class, evaluate_model
import src.utils.dividemix_utils as dataloader
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax




######################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns


######################################################################################################

def co_teaching_loss(model1_loss, model2_loss, rt):
    _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
    _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).cuda()
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss = (model1_loss_filter * model1_loss).sum()

    model2_loss_filter = torch.zeros((model2_loss.size(0))).cuda()
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).sum()

    return model1_loss, model2_loss


def train_step(data_loader, model_list: list, optimizer, optimizer1, optimizer2, criterion, rt,fit=None,
               p_threshold=None, normalization=None,use_fine=False,use_projection=False,epoch=0,args=None):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.
    if_get_feature = True
    model1, model2 = model_list
    model1 = model1.train()
    model2 = model2.train()

    for batch_idx,(x, y_hat,x_idx,mask_train) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)
        y = y_hat

        out1 = model1(x)
        out2 = model2(x)

        model1_loss = criterion(out1, y_hat)
        model2_loss = criterion(out2, y_hat)

        if epoch>args.warmup:
            model1_loss, model2_loss = co_teaching_loss(model1_loss=model1_loss, model2_loss=model2_loss, rt=rt)
        else:
            model1_loss = model1_loss.sum()
            model2_loss = model2_loss.sum()
        # loss exchange
        optimizer1.zero_grad()
        model1_loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 5.0)
        optimizer1.step()

        optimizer2.zero_grad()
        model2_loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
        optimizer2.step()

        avg_loss += (model1_loss.item() + model2_loss.item())

        # Compute accuracy
        acc = torch.eq(torch.argmax(out1, 1), y).float()
        avg_accuracy += acc.mean()
        global_step += 1

    return avg_accuracy / global_step, avg_loss / global_step, [model1, model2]


def test_step(data_loader, model1,model2):
    model1 = model1.eval()
    model2= model2.eval()
    yhat = []
    ytrue = []
    # total_num = len(data_loader.dataset.tensors[1])
    # avg_accuracy = 0.
    if len(data_loader.sampler.data_source.tensors)==4:
        for x, y, _, clean in data_loader:
            x, y = x.to(device), y.to(device)
            logits1 = model1(x)
            logits2 = model2(x)
            logits = (logits1 + logits2) / 2
            yhat.append(logits.detach().cpu().numpy())
            ytrue.append(clean)
            # acc = torch.eq(torch.argmax(logits, 1), y)
            # acc = acc.cpu().numpy()
            # acc = np.sum(acc)
            # avg_accuracy += acc

    else:
        for x, y in data_loader:
            x = x.to(device)
            logits1 = model1(x)
            logits2 = model2(x)
            logits = (logits1 + logits2) / 2
            yhat.append(logits.detach().cpu().numpy())
            ytrue.append(y)
            # acc = torch.eq(torch.argmax(logits, 1), y)
            # acc = acc.cpu().numpy()
            # acc = np.sum(acc)
            # avg_accuracy += acc
    yhat = np.concatenate(yhat, axis=0)
    ytrue = np.concatenate(ytrue,axis=0)
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)
    accuracy = accuracy_score(ytrue, y_hat_labels)
    f1_weighted = f1_score(ytrue, y_hat_labels, average='weighted')

    return accuracy, f1_weighted


def valid_step(data_loader, model):
    model = model.eval()
    total_num = len(data_loader.datasets.tensors[1])
    avg_accuracy = 0.

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        acc = torch.eq(torch.argmax(logits, 1), y)
        acc = acc.cpu().numpy()
        acc = np.sum(acc)
        avg_accuracy += acc
    return avg_accuracy / total_num


def update_reduce_step(cur_step, num_gradual, tau=0.5,args=None):
    # if cur_step > args.warmup:
    return 1.0 - tau * min((cur_step) / num_gradual, 1)
    # else:
    #     return 1.0


def train_model(models, args, train_dataset,test_dataset,test_loader_ori=None,train_loader_ori=None,saver=None):
    model1, model2 = models
    loader = dataloader.dividemix_dataloader(args.dataset, r=args.r, noise_mode=args.label_noise, batch_size=args.batch_size,
                                         num_workers=args.num_workers,train_dataset=train_dataset,test_dataset=test_dataset,
                                             sample_len=args.sample_len)

    # optimizer = optim.Adam(chain(model1.parameters(), model2.parameters()), lr=args.lr, eps=1e-4)
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, eps=1e-4)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, eps=1e-4)

    all_loss = [[], []]  # save the history of losses from two networks
    train_accuracy_list=[]
    test_acc_list = []
    f1s = []
    try:
        for e in range(args.epochs):
            test_loader = loader.run('test')
            eval_loader = loader.run('eval_train')

            if e < args.warmup:
                warmup_trainloader = loader.run('warmup')
                print('Warmup Net1')
                dividemix_warmup(e, model1, optimizer1, warmup_trainloader,args=args)
                print('\nWarmup Net2')
                dividemix_warmup(e, model2, optimizer2, warmup_trainloader,args=args)
            else:
                prob1, all_loss[0] = eval_train(model1, all_loss[0],eval_loader,args=args)
                prob2, all_loss[1] = eval_train(model2, all_loss[1],eval_loader,args=args)
                avg_loss= (sum(all_loss[0])+sum(all_loss[1]))/2
                pred1 = (prob1 > args.p_threshold)
                pred2 = (prob2 > args.p_threshold)
                if pred1.sum()==0:
                    pred1=(prob1>=prob1.mean())
                if pred2.sum()==0:
                    pred2=(prob2>=prob2.mean())

                print('Train Net1')
                labeled_trainloader, unlabeled_trainloader = loader.run('train', pred2, prob2)  # co-divide
                ################################### when dividemix ###############################
                if unlabeled_trainloader==None:
                    model1 = dividemix_train_nounlabel(e, model1, model2, optimizer1, labeled_trainloader,
                                                       args=args)  # train net1
                else:
                    model1 = dividemix_train(e, model1, model2, optimizer1, labeled_trainloader, unlabeled_trainloader,
                                             args=args)  # train net1
                ##################################################################################

                ################### when dividemix w/o unlabel samples ###############################
                # model1 = dividemix_train_nounlabel(e, model1, model2, optimizer1, labeled_trainloader,
                #                                    args=args)  # train net1
                #######################################################################################

                print('\nTrain Net2')
                ################################### when dividemix ###############################
                labeled_trainloader, unlabeled_trainloader = loader.run('train', pred1, prob1)  # co-divide
                if unlabeled_trainloader==None:
                    model2 = dividemix_train_nounlabel(e, model2, model1, optimizer2, labeled_trainloader,
                                                       args=args)  # train net2
                else:
                    model2 = dividemix_train(e, model2, model1, optimizer2, labeled_trainloader, unlabeled_trainloader,
                                             args=args)  # train net2
                ##################################################################################

                ################### when dividemix w/o unlabel samples ###############################
                # model2 = dividemix_train_nounlabel(e, model2, model1, optimizer2, labeled_trainloader,
                #                                    args=args)  # train net2
                #######################################################################################

            # testing/valid step
            train_accuracy, _ = test_step(data_loader=train_loader_ori,
                                      model1=model1,model2=model2)
            test_accuracy, f1 = test_step(data_loader=test_loader_ori,
                                      model1=model1,model2=model2)
            test_acc_list.append(test_accuracy)
            f1s.append(f1)

            print(
                '\nepoch {} - \tTest accuracy {:.4f}'.format(
                    e + 1,
                    test_accuracy))
            train_accuracy_list.append(train_accuracy)
    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    if args.plt_loss_hist:
        plot_train_loss_and_test_acc(test_acc_list,train_accuracy,args=args,
                                     saver=saver,save=True)
    training_results = dict()
    training_results['max_valid_acc'] = np.max(test_acc_list)
    training_results['max_valid_acc_epoch'] = np.argmax(test_acc_list)
    training_results['avg_last_ten_valid_acc'] = np.mean(test_acc_list[-10:])
    training_results['max_valid_f1'] = np.max(f1s)
    training_results['max_valid_f1_epoch'] = np.argmax(f1s)
    training_results['avg_last_ten_valid_f1'] = np.mean(f1s[-10:])
    return model1, training_results


def boxplot_results(data, keys, classes, network, saver):
    n = len(keys)
    x = 'noise'
    hue = 'correct'
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 7 + (n * 0.1)), sharex='all')
    for ax, k in zip(axes, keys):
        sns.boxplot(x=x, y=k, hue=hue, data=data, ax=ax)
        ax.grid(True, alpha=0.2)
    axes[0].set_title('Model:{} classes:{} - Co-Teaching'.format(network, classes))
    fig.tight_layout()
    saver.save_fig(fig, 'boxplot')


def train_eval_model(model, x_train, x_valid, x_test, Y_train, Y_valid, Y_test, Y_train_clean, Y_valid_clean,
                     ni, args, saver, plt_embedding=False, plt_cm=False,mask_train=None):

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))), torch.from_numpy(Y_train_clean))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(Y_valid).long())
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                   num_workers=args.num_workers)

    ######################################################################################################
    # Train model
    model, training_results = train_model(model, args, train_dataset=train_dataset,test_dataset=test_dataset,
                                          test_loader_ori=test_loader,train_loader_ori=train_eval_loader,saver=saver)
    print('Train ended')
    print("training_results = ", training_results)

    ######################################################################################################
    train_results = evaluate_class(model, x_train, Y_train, Y_train_clean, train_eval_loader, ni, saver,
                                   'CNN', 'Train', True, plt_cm=plt_cm, plt_lables=False)
    valid_results = evaluate_class(model, x_valid, Y_valid, Y_valid_clean, valid_loader, ni, saver,
                                   'CNN', 'Valid', True, plt_cm=plt_cm, plt_lables=False)
    test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
                                  'Test', True, plt_cm=plt_cm, plt_lables=False)

    test_results['max_valid_acc'] = training_results['max_valid_acc']
    test_results['max_valid_acc_epoch'] = training_results['max_valid_acc_epoch']
    test_results['avg_last_ten_valid_acc'] = training_results['avg_last_ten_valid_acc']
    test_results['max_valid_f1'] = training_results['max_valid_f1']
    test_results['max_valid_f1_epoch'] = training_results['max_valid_f1_epoch']
    test_results['avg_last_ten_valid_f1'] = training_results['avg_last_ten_valid_f1']



    if plt_embedding and args.embedding_size <= 3:
        plot_embedding(model.encoder, train_eval_loader, valid_loader, Y_train_clean, Y_valid_clean,
                       Y_train, Y_valid, network='CNN', saver=saver, correct=True)



    plt.close('all')
    torch.cuda.empty_cache()
    return train_results, valid_results, test_results


def main_wrapper_dividemix(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean, saver,seeds=[0]):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)
            self.args = args
            self.path = path
            self.makedir_()
            # self.make_log()

    classes = len(np.unique(Y_train_clean))
    args.nbins = classes
    history = x_train.shape[1]

    # Network definition
    classifier1 = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                   norm=args.normalization)
    classifier2 = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                   norm=args.normalization)

    model1 = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                   seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                   padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)
    model2 = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                   seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                   padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model1 = MetaModel(ae=model1, classifier=classifier1, name='CNN').to(device)
    model2 = MetaModel(ae=model2, classifier=classifier2, name='CNN').to(device)
    models = [model1, model2]

    nParams = sum([p.nelement() for p in model1.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % ('CNN', readable(nParams))
    print(s)
    saver.append_str([s])

    ######################################################################################################
    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train_clean.shape, [(Y_train_clean == i).sum() for i in np.unique(Y_train_clean)])
    print('Validation:', x_valid.shape, Y_valid_clean.shape,
          [(Y_valid_clean == i).sum() for i in np.unique(Y_valid_clean)])
    print('Test:', x_test.shape, Y_test_clean.shape, [(Y_test_clean == i).sum() for i in np.unique(Y_test_clean)])
    saver.append_str(['Train: {}'.format(x_train.shape), 'Validation:{}'.format(x_valid.shape),
                      'Test: {}'.format(x_test.shape), '\r\n'])

    ######################################################################################################
    # Main loop
    df_results = pd.DataFrame()
    # seeds = np.random.choice(1000, args.n_runs, replace=False)
    seeds = seeds
    print("seeds = ", seeds)

    for run, seed in enumerate(seeds):
        print()
        print('#' * shutil.get_terminal_size().columns)
        print('EXPERIMENT: {}/{} -- RANDOM SEED:{}'.format(run + 1, args.n_runs, seed).center(columns))
        print('#' * shutil.get_terminal_size().columns)
        print()

        args.seed = seed

        reset_seed_(seed)
        models = [reset_model(m) for m in models]
        # torch.save(model.state_dict(), os.path.join(saver.path, 'initial_weight.pt'))

        saver_loop = SaverSlave(os.path.join(saver.path, f'seed_{seed}'))

        i = 0
        for ni in args.ni:
            saver_slave = SaverSlave(os.path.join(saver.path, f'seed_{seed}', f'ratio_{ni}'))
            i += 1
            # True or false
            print('+' * shutil.get_terminal_size().columns)
            print('HyperRun: %d/%d' % (i, len(args.ni)))
            print('Label noise ratio: %.3f' % ni)
            print('+' * shutil.get_terminal_size().columns)
            # saver.append_str(['#' * 100, 'Label noise ratio: %f' % ni])

            reset_seed_(seed)
            models = [reset_model(m) for m in models]

            Y_train, mask_train = flip_label(x_train,Y_train_clean, ni, args)
            Y_valid, mask_valid = flip_label(x_valid,Y_valid_clean, ni, args)
            Y_test = Y_test_clean

            train_results, valid_results, test_results = train_eval_model(models, x_train, x_valid, x_test, Y_train,
                                                                          Y_valid, Y_test, Y_train_clean,
                                                                          Y_valid_clean,
                                                                          ni, args, saver_slave,
                                                                          plt_embedding=args.plt_embedding,
                                                                          plt_cm=args.plt_cm,
                                                                          mask_train=mask_train)

            keys = list(test_results.keys())
            test_results['noise'] = ni
            test_results['seed'] = seed
            test_results['correct'] = 'Co-teaching'
            test_results['losses'] = map_abg([0, 1, 0])
            # saver_loop.append_str(['Test Results:'])
            # saver_loop.append_dict(test_results)
            df_results.append(test_results, ignore_index=True)

        if args.plt_cm:
            fig_title = f"CO-TEACHING -- Dataset: {args.dataset} - Model: {'CNN'} - classes:{classes} - runs:{args.n_runs} "
            plot_results(df_results.loc[df_results.seed == seed], keys, saver_loop, x='noise', hue='correct',
                         col='losses',
                         kind='bar', style='whitegrid', title=fig_title)
    if args.plt_cm:
        # Losses column should  not change here
        fig_title = f"CO-TEACHING -- Dataset: {args.dataset} - Model: {'CNN'} - classes:{classes} - runs:{args.n_runs} "
        plot_results(df_results, keys, saver, x='noise', hue='correct', col='losses', kind='box', style='whitegrid',
                     title=fig_title)

    # boxplot_results(df_results, keys, classes, 'CNN')
    # results_summary = df_results.groupby(['noise', 'correct'])[keys].describe().T
    # saver.append_str(['Results main summary', results_summary])

    remove_empty_dirs(saver.path)

    return test_results


def plot_train_loss_and_test_acc(test_acc_list,Train_acc_list,args=None,saver=None,save=False):
    fig = plt.figure(figsize=(1200,800))

    plt.gcf().set_facecolor(np.ones(3) * 240 / 255)
    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # l1,l2,l3,l4 = None,None,None,None
    l2 = ax1.plot(test_acc_list, '-', c='blue', label='Test acc', linewidth=1)
    l3 = ax1.plot(Train_acc_list,'-',c='green',label='Train acc',linewidth=1)

    lns = l2 + l3
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs,loc='upper right')
    # plt.legend(handles=[l1,l2],labels=["Training loss","Test acc"],loc='upper right')

    plt.axvline(args.warmup,color='g',linestyle='--')

    ax1.set_xlabel('epoch',  size=18)
    ax1.set_ylabel('Acc',size=18)
    # ax2.set_ylabel('Test acc',  size=18)
    plt.gcf().autofmt_xdate()
    plt.title(f'Model:new model dataset:{args.dataset}')
    plt.grid(True)

    plt.tight_layout()

    saver.save_fig(fig, name=args.dataset)


def warmup_co_teaching(data_loader, model, optimizer, criterion):

    model = model.train()
    for batch_idx,(x, y_hat,x_idx,batch_mask) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)

        h = model.encoder1(x).squeeze()
        y = model.classifier1(h)

        model_loss_1 = criterion(y, y_hat)
        model_loss_1 = model_loss_1.sum()
        model_loss = model_loss_1
        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    return model

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up,args):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up,args=args)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def linear_rampup(current, warm_up, rampup_length=16,args=None):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)


def eval_train(model, all_loss,eval_loader,args):
    model.eval()
    losses = torch.zeros(args.num_training_samples)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss(reduction='none')(outputs, targets)
            # for b in range(inputs.size(0)):
            #     losses[index[b]] = loss[b]
            losses[index] = loss.cpu()
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if args.r == 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    # gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss

def dividemix_warmup(epoch,net,optimizer,dataloader,args):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, _,_) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        if args.label_noise==1 or args.label_noise==-1:  # penalize confident prediction for asymmetric noise
            penalty = NegEntropy()(outputs)
            L = loss + penalty
        elif args.label_noise==0:
            L = loss
        L.backward()
        optimizer.step()


def dividemix_train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader,args):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.nbins).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        # inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda().float(), inputs_x2.cuda().float(), labels_x.cuda(), w_x.cuda()
        # inputs_u, inputs_u2 = inputs_u.cuda().float(), inputs_u2.cuda().float()
        labels_x, w_x = labels_x.cuda(), w_x.cuda()
        inputs_x = torch.from_numpy(
            tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                inputs_x.cpu().numpy())).float().to(
            device)
        inputs_x2 = torch.from_numpy(
            tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                inputs_x2.cpu().numpy())).float().to(
            device)
        inputs_u = torch.from_numpy(
            tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                inputs_u.cpu().numpy())).float().to(
            device)
        inputs_u2 = torch.from_numpy(
            tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                inputs_u2.cpu().numpy())).float().to(
            device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            # if  len(inputs_u)==1 or len(inputs_u2)==1 or len(inputs_u2)==1:
            #     continue
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
                                                                                                        dim=1) + torch.softmax(
                outputs_u22, dim=1)) / 4
            ptu = pu ** (1 / args.T)  # temparature sharpening

            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        # mixed_input = all_inputs
        # mixed_target = all_targets

        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx, Lu, lamb = SemiLoss()(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, args.warmup, args=args)

        # regularization
        prior = torch.ones(args.nbins) / args.nbins
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%d | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                         % (args.dataset, args.r, args.label_noise, epoch, args.epochs, batch_idx + 1, num_iter,
                            Lx.item(), Lu.item()))
        sys.stdout.flush()
        return net

def dividemix_train_nounlabel(epoch, net, net2, optimizer, labeled_trainloader,args):
    net.train()
    net2.eval()  # fix one network and train the other

    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.nbins).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        # inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda().float(), inputs_x2.cuda().float(), labels_x.cuda(), w_x.cuda()
        labels_x, w_x = labels_x.cuda(), w_x.cuda()
        inputs_x = torch.from_numpy(
            tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                inputs_x.cpu().numpy())).float().to(
            device)
        inputs_x2 = torch.from_numpy(
            tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                inputs_x2.cpu().numpy())).float().to(
            device)
        with torch.no_grad():

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x * labels_x + (1 - w_x) * px
            ptx = px ** (1 / args.T)  # temparature sharpening

            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2], dim=0)
        all_targets = torch.cat([targets_x, targets_x], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        # input_a, input_b = all_inputs, all_inputs[idx]
        # target_a, target_b = all_targets, all_targets[idx]

        # mixed_input = l * input_a + (1 - l) * input_b
        # mixed_target = l * target_a + (1 - l) * target_b
        mixed_input = all_inputs
        mixed_target = all_targets

        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx, Lu, lamb = SemiLoss()(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, args.warmup, args=args)

        # regularization
        prior = torch.ones(args.nbins) / args.nbins
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%d | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f Size/Batch_size:%d/%d'
                         % (args.dataset, args.r, args.label_noise, epoch, args.epochs, batch_idx + 1, num_iter,
                            Lx.item(), Lu.item(),args.num_training_samples,batch_size))
        sys.stdout.flush()
        return net
