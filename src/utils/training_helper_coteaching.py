import collections
import os
import random
import shutil
from itertools import chain
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
# torch.cuda.set_device(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tsaug
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax

from src.models.MultiTaskClassification import MetaModel, NonLinClassifier
from src.models.model import CNNAE
from src.utils.plotting_utils import plot_results, plot_embedding
from src.utils.saver import Saver
from src.utils.utils import readable, reset_seed_, reset_model, flip_label, map_abg, remove_empty_dirs, \
    evaluate_class, hyperbolic_tangent, sigua_loss


######################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns


######################################################################################################

def co_teaching_loss(model1_loss, model2_loss, rt,loss_all1=None,loss_all2=None,args=None,epoch=None,x_idxs=None):
    if loss_all1 is None:
        _, model1_sm_idx = torch.topk(model1_loss, k=int(int(model1_loss.size(0)) * rt), largest=False)
        _, model2_sm_idx = torch.topk(model2_loss, k=int(int(model2_loss.size(0)) * rt), largest=False)
    else:
        gamma = args.gamma
        if args.mean_loss_len>args.warmup:
            loss_mean1 = gamma * loss_all1[x_idxs, epoch] + (1 - gamma) * (
                loss_all1[x_idxs, (epoch - args.warmup + 1):epoch].mean(axis=1))
            loss_mean2 = gamma * loss_all2[x_idxs, epoch] + (1 - gamma) * (
                loss_all2[x_idxs, (epoch - args.warmup + 1):epoch].mean(axis=1))
        else:
            loss_mean1 = gamma * loss_all1[x_idxs, epoch] + (1 - gamma) * (
                loss_all1[x_idxs, (epoch - args.mean_loss_len + 1):epoch].mean(axis=1))
            loss_mean2 = gamma * loss_all2[x_idxs, epoch] + (1 - gamma) * (
                loss_all2[x_idxs, (epoch - args.mean_loss_len + 1):epoch].mean(axis=1))

        _, model1_sm_idx = torch.topk(torch.from_numpy(loss_mean1), k=int(int(model1_loss.size(0)) * rt), largest=False)
        _, model2_sm_idx = torch.topk(torch.from_numpy(loss_mean2), k=int(int(model2_loss.size(0)) * rt), largest=False)

    # co-teaching
    model1_loss_filter = torch.zeros((model1_loss.size(0))).to(device)
    model1_loss_filter[model2_sm_idx] = 1.0
    model1_loss = (model1_loss_filter * model1_loss).sum()

    model2_loss_filter = torch.zeros((model2_loss.size(0))).to(device)
    model2_loss_filter[model1_sm_idx] = 1.0
    model2_loss = (model2_loss_filter * model2_loss).sum()

    return model1_loss, model2_loss, model1_sm_idx, model2_sm_idx


def train_step(data_loader, model_list: list, optimizer, optimizer1, optimizer2, criterion, rt,fit=None,
               p_threshold=None, normalization=None,epoch=0,args=None):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.
    if_get_feature = True
    model1, model2 = model_list
    model1 = model1.train()
    model2 = model2.train()

    for batch_idx,(x, y_hat,x_idx,_) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)
        y = y_hat

        out1 = model1(x)
        out2 = model2(x)

        model1_loss = criterion(out1, y_hat)
        model2_loss = criterion(out2, y_hat)

        if epoch > args.warmup:
            model1_loss, model2_loss ,_,_= co_teaching_loss(model1_loss=model1_loss, model2_loss=model2_loss, rt=rt)
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

        avg_loss += (model1_loss.item()+model2_loss.item())

        # Compute accuracy
        acc = torch.eq(torch.argmax(out1, 1), y).float()
        avg_accuracy += acc.mean().cpu().numpy()
        global_step += 1

    return avg_accuracy / global_step, avg_loss / global_step, [model1, model2]


def test_step(data_loader, model1,model2=None):
    model1 = model1.eval()
    if model2 is not None:
        model2 = model2.eval()

    yhat = []
    ytrue = []

    for x, y in data_loader:
        x = x.to(device)
        if model2 is not None:
            logits1 = model1(x)
            logits2 = model2(x)
            logits = (logits1 + logits2) / 2
        else:
            logits = model1(x)

        yhat.append(logits.detach().cpu().numpy())
        try:
            y = y.cpu().numpy()
        except:
            y = y.numpy()
        ytrue.append(y)

    yhat = np.concatenate(yhat, axis=0)
    ytrue = np.concatenate(ytrue, axis=0)
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)
    accuracy = accuracy_score(ytrue, y_hat_labels)
    f1_weighted = f1_score(ytrue, y_hat_labels, average='weighted')

    return accuracy, f1_weighted

def update_reduce_step(cur_step, num_gradual, tau=0.5,args=None):
    # if cur_step > args.warmup:
    return 1.0 - tau * min((cur_step) / num_gradual, 1)
    # else:
    #     return 1.0


def train_model(models, train_loader, test_loader, args, tau,saver=None):
    model1, model2 = models
    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(chain(model1.parameters(), model2.parameters()), lr=args.lr, eps=1e-4)
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, eps=1e-4)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, eps=1e-4)
    # learning history
    train_acc_list = []
    train_avg_loss_list = []
    test_acc_list = []
    test_f1s = []
    try:
        loss_all1 = np.zeros((args.num_training_samples, args.epochs))
        loss_all2 = np.zeros((args.num_training_samples, args.epochs))
        for e in range(args.epochs):
            # update reduce step
            rt = update_reduce_step(cur_step=e, num_gradual=args.num_gradual, tau=tau,args=args)

            # training step

            if e <= args.warmup:
                train_accuracy, avg_loss, model_list = warm_up(data_loader1=train_loader,
                                                                            data_loader2=train_loader,
                                                                            model_list=[model1, model2],
                                                                            optimizer1=optimizer1,
                                                                            optimizer2=optimizer2,
                                                                            criterion=criterion,
                                                                            epoch=e,
                                                                            loss_all1=loss_all1,
                                                                            loss_all2=loss_all2)
            else:
                if args.model in ['co_teaching']:
                    train_accuracy, avg_loss, model_list = train_step(data_loader=train_loader,
                                                                      model_list=[model1, model2],
                                                                      optimizer=optimizer,
                                                                      optimizer1=optimizer1,
                                                                      optimizer2=optimizer2,
                                                                      criterion=criterion,
                                                                      rt=rt,
                                                                      epoch=e,
                                                                      args=args)

                elif args.model in ['sigua']:
                    if e==args.warmup+1:
                        last_sel_as_good=None
                    train_accuracy, avg_loss, model_list, current_sel_as_good_epoch = train_sigua_step(
                        data_loader=train_loader,
                        model=model1,
                        optimizer=optimizer1,
                        criterion=criterion,
                        rt=rt,
                        bad_weight=0.001,
                        last_sel_id=last_sel_as_good,
                    args=args)
                    last_sel_as_good=current_sel_as_good_epoch

                elif args.model in ['co_teaching_mloss']:
                    train_accuracy, avg_loss, model_list = train_step_cotea_mloss(data_loader=train_loader,
                                                                               model_list=[model1, model2],
                                                                               optimizer=optimizer,
                                                                               optimizer1=optimizer1,
                                                                               optimizer2=optimizer2,
                                                                               loss_all1=loss_all1,
                                                                               loss_all2=loss_all2,
                                                                               criterion=criterion,
                                                                               rt=rt,
                                                                               epoch=e,
                                                                               args=args)


            model1, model2 = model_list

            # testing
            test_accuracy, f1 = test_step(data_loader=test_loader,
                                      model1=model1,model2=model2)

            train_acc_list.append(train_accuracy)
            train_avg_loss_list.append(avg_loss)
            test_acc_list.append(test_accuracy)
            test_f1s.append(f1)

            print(
                '{} epoch - Train Loss {:.4f}\tTrain accuracy {:.4f}\tTest accuracy {:.4f}\tReduce rate {:.4f}'.format(
                    e + 1,
                    avg_loss,
                    train_accuracy,
                    test_accuracy,
                    rt))

    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    if args.plt_loss_hist:
        plot_train_loss_and_test_acc(train_avg_loss_list,test_acc_list,args,pred_precision=train_acc_list,
                                     saver=saver,save=True)

    # we test the final model at line 253
    test_results_last_ten_epochs = dict()
    test_results_last_ten_epochs['last_ten_test_acc'] = test_acc_list[-10:]
    test_results_last_ten_epochs['last_ten_test_f1'] = test_f1s[-10:]
    return model_list, test_results_last_ten_epochs

def train_eval_model(model, x_train, x_test, Y_train, Y_test, Y_train_clean,
                     ni, args, saver, plt_embedding=True, plt_cm=True,mask_train=None):

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))), torch.from_numpy(Y_train_clean)) # 'Y_train_clean' is used for evaluation instead of training.
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers)


    ######################################################################################################
    # Train model

    model, test_results_last_ten_epochs = train_model(model, train_loader, test_loader, args, ni,saver=saver)
    print('Train ended')

    ########################################## Eval ############################################

    # save test_results: test_acc(the final model), test_f1(the final model), avg_last_ten_test_acc, avg_last_ten_test_f1
    test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
                                  'Test', True, plt_cm=plt_cm, plt_lables=False)
    test_results['avg_last_ten_test_acc'] = np.mean(test_results_last_ten_epochs['last_ten_test_acc'])
    test_results['avg_last_ten_test_f1'] = np.mean(test_results_last_ten_epochs['last_ten_test_f1'])

    torch.cuda.empty_cache()
    return test_results


def main_wrapper(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=None):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)

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
    print('Test:', x_test.shape, Y_test_clean.shape, [(Y_test_clean == i).sum() for i in np.unique(Y_test_clean)])
    saver.append_str(['Train: {}'.format(x_train.shape),
                      'Test: {}'.format(x_test.shape), '\r\n'])

    ######################################################################################################
    # Main loop
    df_results = pd.DataFrame()
    if seed is None:
        seed = np.random.choice(1000, 1, replace=False)

    print()
    print('#' * shutil.get_terminal_size().columns)
    print('RANDOM SEED:{}'.format(seed).center(columns))
    print('#' * shutil.get_terminal_size().columns)
    print()

    args.seed = seed

    i = 0
    ni = args.ni
    saver_slave = SaverSlave(os.path.join(saver.path, f'seed_{seed}', f'ratio_{ni}'))
    i += 1
    # True or false
    print('+' * shutil.get_terminal_size().columns)
    print('Label noise ratio: %.3f' % ni)
    print('+' * shutil.get_terminal_size().columns)
    # saver.append_str(['#' * 100, 'Label noise ratio: %f' % ni])

    reset_seed_(seed)
    models = [reset_model(m) for m in models]

    Y_train, mask_train = flip_label(x_train, Y_train_clean, ni, args)
    Y_test = Y_test_clean

    test_results = train_eval_model(models, x_train, x_test, Y_train, Y_test, Y_train_clean,
                                    ni, args, saver_slave,
                                    plt_embedding=args.plt_embedding,
                                    plt_cm=args.plt_cm,
                                    mask_train=mask_train)

    remove_empty_dirs(saver.path)

    return test_results

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

def warm_up(data_loader1,data_loader2, model_list: list, optimizer1, optimizer2, criterion,epoch=None,
                         loss_all1=None,loss_all2=None):
    global_step = 0
    avg_accuracy = 0.
    avg_loss1 = 0.
    avg_loss2 = 0.
    # if_get_feature = True
    model1, model2 = model_list
    model1 = model1.train()
    model2 = model2.train()

    for batch_idx,(x1, y_hat1,x_idx1,_) in enumerate(data_loader1):
        if x1.shape[0]==1:
            continue
        # Forward and Backward propagation
        x1, y_hat1 = x1.to(device), y_hat1.to(device)
        y1 = y_hat1

        if hasattr(model1,'decoder'):
            h1=model1.encoder(x1)
            h1d=model1.decoder(h1)
            out1=model1.classifier(h1.squeeze(-1))
            model1_loss = criterion(out1, y_hat1)
            loss_all1[x_idx1, epoch] = model1_loss.data.detach().clone().cpu().numpy()
            model1_loss = model1_loss.sum()+nn.MSELoss(reduction='mean')(h1d,x1)
        else:
            out1 = model1(x1)
            model1_loss = criterion(out1, y_hat1).sum()

        ############################################################################################################

        # loss exchange
        optimizer1.zero_grad()
        model1_loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 5.0)
        optimizer1.step()

        avg_loss1 += model1_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out1, 1), y1).float()
        avg_accuracy += acc.mean().cpu().numpy()
        global_step += 1

    for batch_idx,(x2, y_hat2,x_idx2,_) in enumerate(data_loader2):
        if x2.shape[0]==1:
            continue
        # Forward and Backward propagation
        x2, y_hat2 = x2.to(device), y_hat2.to(device)

        if hasattr(model2,'decoder'):
            h2=model2.encoder(x2)
            h2d=model2.decoder(h2)
            out2=model2.classifier(h2.squeeze(-1))
            model2_loss = criterion(out2, y_hat2)
            loss_all2[x_idx2, epoch] = model2_loss.data.detach().clone().cpu().numpy()
            model2_loss=model2_loss.sum()+nn.MSELoss(reduction='mean')(h2d,x2)
        else:
            out2 = model2(x2)
            model2_loss = criterion(out2, y_hat2).sum()

        ############################################################################################################

        # loss exchange
        optimizer2.zero_grad()
        model2_loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
        optimizer2.step()

        avg_loss2 += model2_loss.item()

    avg_loss=(avg_loss1+avg_loss2)/2

    return avg_accuracy / global_step, avg_loss / global_step, [model1, model2]


def train_step_cotea_mloss(data_loader, model_list: list, optimizer, optimizer1, optimizer2, criterion,
                        loss_all1=None,loss_all2=None,rt=None,fit=None,p_threshold=None, normalization=None,
                        epoch=0,args=None):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.
    if_get_feature = True
    model1, model2 = model_list
    model1 = model1.train()
    model2 = model2.train()

    for batch_idx,(x, y_hat,x_idx,_) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)
        y = y_hat

        h1 = model1.encoder(x)
        h2 = model2.encoder(x)
        out1 = model1.classifier(h1.squeeze(-1))
        out2 = model2.classifier(h2.squeeze(-1))

        model1_loss = criterion(out1, y_hat)
        model2_loss = criterion(out2, y_hat)

        if loss_all1 is not None:
            loss_all1[x_idx,epoch]=model1_loss.data.detach().clone().cpu().numpy()
            loss_all2[x_idx,epoch]=model2_loss.data.detach().clone().cpu().numpy()

        model1_loss, model2_loss,_,_ = co_teaching_loss(model1_loss=model1_loss, model2_loss=model2_loss, rt=rt,
                                                        loss_all1=loss_all1,loss_all2=loss_all2,args=args,epoch=epoch,
                                                        x_idxs=x_idx)

        # loss exchange
        optimizer1.zero_grad()
        model1_loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), 5.0)
        optimizer1.step()

        optimizer2.zero_grad()
        model2_loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
        optimizer2.step()

        avg_loss += (model1_loss.item() + model2_loss.item())/2

        # Compute accuracy
        acc = torch.eq(torch.argmax(out1, 1), y).float()
        avg_accuracy += acc.mean().cpu().numpy()
        global_step += 1

    return avg_accuracy / global_step, avg_loss / global_step, [model1, model2]

def train_sigua_step(data_loader, model, optimizer, criterion, rt, bad_weight=0.001,last_sel_id=None,args=None):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.
    model = model.train()
    current_sel_as_good_epoch=np.array([])
    for x, y_hat,x_idx,_ in data_loader:
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)
        y = y_hat

        out = model(x)

        model_loss = criterion(out, y_hat)

        model_loss, current_sel_as_good_batch = sigua_loss(model_loss=model_loss, rt=rt, bad_weight=bad_weight,
                                                               last_sel_id=last_sel_id,
                                                               current_batch_idx=x_idx,args=args)

        np.concatenate((current_sel_as_good_epoch,current_sel_as_good_batch))
        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out, 1), y).float()
        avg_accuracy += acc.mean().cpu().numpy()
        global_step += 1

    return avg_accuracy / global_step, avg_loss / global_step, [model,None],current_sel_as_good_epoch


