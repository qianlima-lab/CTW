import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# torch.cuda.set_device(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tsaug
import time
from sklearn.metrics import accuracy_score, f1_score
from scipy.special import softmax

from src.models.MultiTaskClassification import NonLinClassifier, MetaModel_AE
from src.models.model import CNNAE
from src.utils.saver import Saver
from src.utils.utils import readable, reset_seed_, reset_model, flip_label, remove_empty_dirs, \
    evaluate_class, to_one_hot,small_loss_criterion_EPS
from src.plot.visualization import t_sne,t_sne_during_train

######################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns


######################################################################################################

def add_to_confident_set_id(args=None,confident_set_id=None,train_dataset=None,epoch=None,conf_num=None):

    xs, ys, _, y_clean = train_dataset.tensors
    ys=ys.cpu().numpy()
    y_clean=y_clean.cpu().numpy()
    TP_all=0
    FP_all=0
    for i in range(args.nbins):
        confnum_row = dict()
        confnum_row['epoch'] = epoch
        if args.sel_method == 3:
            confnum_row['method']='Our model'
        else: # sel_method in [1,2]
            confnum_row['method'] = 'Class by Class'
        confnum_row['label']=i
        confnum_row['total']=sum(ys[confident_set_id]==i)
        confnum_row['TP'] = sum((y_clean[confident_set_id][ys[confident_set_id]==i]==i))
        TP_all=TP_all+confnum_row['TP']
        confnum_row['FP'] = sum((y_clean[confident_set_id][ys[confident_set_id]==i]!=i))
        FP_all =FP_all+ confnum_row['FP']
        confnum_row['seed'] = args.seed

        conf_num.append(confnum_row)
    estimate_noise_rate=TP_all/(TP_all+FP_all)
    return conf_num, estimate_noise_rate

def save_model_and_sel_dict(model,args,sel_dict=None):
    model_state_dict = model.state_dict()
    datestr = time.strftime(('%Y%m%d'))
    model_to_save_dir = os.path.join(args.basicpath, 'src', 'model_save', args.dataset)
    if not os.path.exists(model_to_save_dir):
        os.makedirs(model_to_save_dir, exist_ok=True)

    if args.label_noise == -1:
        label_noise = 'inst{}'.format(int(args.ni * 100))
    elif args.label_noise == 0:
        label_noise = 'sym{}'.format(int(args.ni * 100))
    else:
        label_noise = 'asym{}'.format(int(args.ni * 100))
    filename = os.path.join(model_to_save_dir, args.model)
    if sel_dict is not None:
        filename_sel_dict = '{}{}_{}_{}_sel_dict.npy'.format(filename, args.aug, label_noise, datestr)
        np.save(filename_sel_dict, sel_dict)  # save sel_ind
    filename = '{}{}_{}_{}.pt'.format(filename, args.aug, label_noise, datestr)
    torch.save(model_state_dict, filename)  # save model

def test_step(data_loader, model,model2=None):
    model = model.eval()
    if model2 is not None:
        model2 = model2.eval()

    yhat = []
    ytrue = []

    for x, y in data_loader:
        x = x.to(device)

        if model2 is not None:
            logits1 = model(x)
            logits2 = model2(x)
            logits = (logits1 + logits2) / 2
        else:
            logits = model(x)

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


def train_model(model, train_loader, test_loader, args,train_dataset=None,saver=None):

    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
    # learning history
    train_acc_list = []
    train_acc_list_aug = []
    train_avg_loss_list = []
    test_acc_list = []
    test_f1s = []
    try:
        loss_all = np.zeros((args.num_training_samples, args.epochs))


        conf_num = []
        for e in range(args.epochs):
            sel_dict = {'sel_ind': [], 'lam': [], 'mix_ind': []}

            # training step
            if e <= args.warmup:
                train_accuracy, avg_loss, model_new = warmup_CTW(data_loader=train_loader,
                                                                            model=model,
                                                                            optimizer=optimizer,
                                                                            criterion=criterion,
                                                                            epoch=e,
                                                                            loss_all=loss_all,
                                                                            args=args)
            else:
                train_accuracy, avg_loss, model_new, confident_set_id = train_step_CTW(
                    data_loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    loss_all=loss_all,
                    criterion=criterion,
                    epoch=e,
                    args=args, sel_dict=sel_dict)

                if args.confcsv is not None: # save confident samples' id to visualize
                    conf_num, _ = add_to_confident_set_id(args=args,
                                                       confident_set_id=confident_set_id.astype(int),
                                                       train_dataset=train_dataset, epoch=e,
                                                       conf_num=conf_num)
            model = model_new

            if args.tsne_during_train and args.seed == args.manual_seeds[0] and e in args.tsne_epochs:
                xs, ys, _, y_clean = train_dataset.tensors
                with torch.no_grad():
                    t_sne_during_train(xs, ys, y_clean, model=model, tsne=True, args=args,sel_dict=sel_dict,epoch=e)

            # testing
            test_accuracy, f1 = test_step(data_loader=test_loader,
                                      model=model)

            # train results each epoch
            train_acc_list.append(train_accuracy[0])
            train_acc_list_aug.append(train_accuracy[1])
            train_acc_oir =train_accuracy[0]
            train_avg_loss_list.append(avg_loss)

            # test results each epoch
            test_acc_list.append(test_accuracy)
            test_f1s.append(f1)

            print(
                '{} epoch - Train Loss {:.4f}\tTrain accuracy {:.4f}\tTest accuracy {:.4f}'.format(
                    e + 1,
                    avg_loss,
                    train_acc_oir,
                    test_accuracy))

    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    if args.confcsv is not None:
        csvpath = os.path.join(args.basicpath, 'src', 'bar_info')
        if not os.path.exists(csvpath):
            os.makedirs(csvpath)
        pd.DataFrame(conf_num).to_csv(os.path.join(csvpath, args.dataset + str(args.sel_method) + args.confcsv),
                                      mode='a', header=True)
    if args.save_model:
        save_model_and_sel_dict()
    if args.plt_loss_hist:
        plot_train_loss_and_test_acc(train_avg_loss_list,test_acc_list,args,pred_precision=train_acc_list,aug_accs=train_acc_list_aug,
                                     saver=saver,save=True)
    if args.plot_tsne and args.seed==args.manual_seeds[0]:
        xs,ys,_,y_clean = train_dataset.tensors
        datestr = time.strftime(('%Y%m%d'))
        with torch.no_grad():
            t_sne(xs, ys, y_clean,model=model, tsne=True, args=args,datestr=datestr,sel_dict=sel_dict)

    # we test the final model at line 231.
    test_results_last_ten_epochs = dict()
    test_results_last_ten_epochs['last_ten_test_acc'] = test_acc_list[-10:]
    test_results_last_ten_epochs['last_ten_test_f1'] = test_f1s[-10:]
    return model, test_results_last_ten_epochs


def train_eval_model(model, x_train, x_test, Y_train, Y_test, Y_train_clean,
                     ni, args, saver, plt_embedding=True, plt_cm=True):

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))), torch.from_numpy(Y_train_clean)) # 'Y_train_clean' is used for evaluation instead of training.

    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers)

    # compute noise prior
    ######################################################################################################
    # Train model

    model, test_results_last_ten_epochs = train_model(model, train_loader, test_loader, args,
                                          train_dataset=train_dataset,saver=saver)
    print('Train ended')

    ########################################## Eval ############################################

    # save test_results: test_acc(the final model), test_f1(the final model), avg_last_ten_test_acc, avg_last_ten_test_f1
    # test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
    #                               'Test', True, plt_cm=plt_cm, plt_lables=False) # evaluate_class will evaluate the final model.
    test_results = dict()
    test_results['acc'] = test_results_last_ten_epochs['last_ten_test_acc'][-1]
    test_results['f1_weighted'] = test_results_last_ten_epochs['last_ten_test_f1'][-1]
    test_results['avg_last_ten_test_acc'] = np.mean(test_results_last_ten_epochs['last_ten_test_acc'])
    test_results['avg_last_ten_test_f1'] = np.mean(test_results_last_ten_epochs['last_ten_test_f1'])

    #############################################################################################
    plt.close('all')
    torch.cuda.empty_cache()
    return test_results


def main_wrapper_CTW(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=None):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)
            self.args=args
            self.path = path
            self.makedir_()
            # self.make_log()

    classes = len(np.unique(Y_train_clean))
    args.nbins = classes

    # Network definition
    classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                   norm=args.normalization)

    model = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                   seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                   padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = MetaModel_AE(ae=model, classifier=classifier, name='CNN').to(device)

    nParams = sum([p.nelement() for p in model.parameters()])
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
    if seed is None:
        seed = np.random.choice(1000, 1, replace=False)

    print('#' * shutil.get_terminal_size().columns)
    print('RANDOM SEED:{}'.format(seed).center(columns))
    print('#' * shutil.get_terminal_size().columns)

    args.seed = seed

    ni = args.ni
    saver_slave = SaverSlave(os.path.join(saver.path, f'seed_{seed}', f'ratio_{ni}'))
    # True or false
    print('+' * shutil.get_terminal_size().columns)
    print('Label noise ratio: %.3f' % ni)
    print('+' * shutil.get_terminal_size().columns)

    reset_seed_(seed)
    model = reset_model(model)

    Y_train, mask_train = flip_label(x_train, Y_train_clean, ni, args)
    Y_test = Y_test_clean

    test_results = train_eval_model(model, x_train, x_test, Y_train,
                                                   Y_test, Y_train_clean,
                                                   ni, args, saver_slave,
                                                   plt_embedding=args.plt_embedding,
                                                   plt_cm=args.plt_cm)
    remove_empty_dirs(saver.path)

    return test_results


def plot_train_loss_and_test_acc(avg_train_losses,test_acc_list,args,pred_precision=None,saver=None,save=False,aug_accs=None):


    plt.gcf().set_facecolor(np.ones(3) * 240 / 255)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    l1 = ax.plot(avg_train_losses,'-', c='orangered', label='Training loss', linewidth=1)
    l2 = ax2.plot(test_acc_list, '-', c='blue', label='Test acc', linewidth=1)
    l3 = ax2.plot(pred_precision,'-',c='green',label='Sample_sel acc',linewidth=1)

    if len(aug_accs)>0:
        l4 = ax2.plot(aug_accs, '-', c='yellow', label='Aug acc', linewidth=1)
        lns = l1 + l2 + l3+l4
    else:
        lns = l1 + l2 + l3

    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs,loc='upper right')
    # plt.legend(handles=[l1,l2],labels=["Training loss","Test acc"],loc='upper right')

    plt.axvline(args.warmup,color='g',linestyle='--')

    ax.set_xlabel('epoch',  size=18)
    ax.set_ylabel('Train loss',size=18)
    ax2.set_ylabel('Test acc',  size=18)
    plt.gcf().autofmt_xdate()
    plt.title(f'Model:new model dataset:{args.dataset}')
    plt.grid(True)

    plt.tight_layout()

    saver.save_fig(fig, name=args.dataset)


def warmup_CTW(data_loader, model, optimizer, criterion,epoch=None,
                         loss_all=None,args=None):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.
    model = model.train()

    for batch_idx,(x, y_hat,x_idx,_) in enumerate(data_loader):
        if x.shape[0]==1:
            continue
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)

        if hasattr(model,'decoder'):
            h=model.encoder(x)
            hd=model.decoder(h)
            out=model.classifier(h.squeeze(-1))
            model_loss = criterion(out, y_hat)
            loss_all[x_idx, epoch] = model_loss.data.detach().clone().cpu().numpy()
            model_loss = model_loss.sum()+nn.MSELoss(reduction='mean')(hd,x)
        else:
            out = model(x)
            model_loss = criterion(out, y_hat).sum()

        ############################################################################################################

        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out, 1), y_hat).float()
        avg_accuracy += acc.sum().cpu().numpy()
        global_step += len(y_hat)

    return (avg_accuracy / global_step,0.), avg_loss / global_step, model



def train_step_CTW(data_loader, model, optimizer,  criterion, loss_all=None,epoch=0,args=None,sel_dict=None):
    global_step = 0
    aug_step = 0
    avg_accuracy = 0.
    avg_accuracy_aug = 0.
    avg_loss = 0.
    model = model.train()
    confident_set_id=np.array([])

    for batch_idx,(x, y_hat,x_idx,_) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)
        y = y_hat

        h = model.encoder(x)
        out = model.classifier(h.squeeze(-1))

        hd = model.decoder(h)
        recon_loss=nn.MSELoss(reduction='mean')(hd, x)

        model_loss = criterion(out, y_hat)

        if loss_all is not None:
            loss_all[x_idx,epoch]=model_loss.data.detach().clone().cpu().numpy()

        model_loss, model_sel_idx = small_loss_criterion_EPS(model_loss=model_loss,
                                                             loss_all=loss_all, args=args,
                                                             epoch=epoch, x_idxs=x_idx, labels=y_hat)

        # data Augmentation after selecting clean samples
        if (batch_idx % args.arg_interval == 0) and len(model_sel_idx)!=1:
            x_aug = torch.from_numpy(
                tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                    x[model_sel_idx].cpu().numpy())).float().to(
                device)
            aug_step += 1
            if len(x_aug)==1: # avoid bugs`
                aug_model_loss = 0.
                avg_accuracy_aug = 0.
            else:
                aug_h = model.encoder(x_aug)
                outx_aug=model.classifier(aug_h.squeeze(-1))
                y_hat_aug = y_hat[model_sel_idx]
                aug_model_loss = criterion(outx_aug, y_hat[model_sel_idx]).sum()
                avg_accuracy_aug += torch.eq(torch.argmax(outx_aug, 1), y_hat_aug).float().sum().cpu().numpy()

            if epoch==args.epochs-1 or epoch in args.tsne_epochs:
                sel_dict['sel_ind'].append(x_idx[model_sel_idx].cpu().numpy())

        else:
            aug_model_loss = 0.
            avg_accuracy_aug = 0.

        model_loss = model_loss + args.L_aug_coef*aug_model_loss+args.L_rec_coef*recon_loss

        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc1 = torch.eq(torch.argmax(out, 1), y).float()
        avg_accuracy += acc1.sum().cpu().numpy()

        global_step += 1

        confident_set_id=np.concatenate((confident_set_id,x_idx[model_sel_idx].cpu().numpy()))

    return (avg_accuracy / global_step, avg_accuracy_aug / aug_step), avg_loss / global_step, model,confident_set_id