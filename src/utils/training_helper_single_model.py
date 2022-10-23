from __future__ import print_function

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


from src.models.MultiTaskClassification import MetaModel, NonLinClassifier, MetaModel_AE
from src.models.model import CNNAE
from src.utils.plotting_utils import plot_results, plot_embedding
from src.utils.saver import Saver
from src.utils.utils import readable, reset_seed_, reset_model, flip_label, map_abg, remove_empty_dirs, \
    evaluate_class, compute_noise_prior, to_one_hot,small_loss_criterion_auto_rate2,\
    small_loss_criterion_without_eliminate,small_loss_criterion_auto_rate1,small_loss_criterion
from src.plot.visualization import t_sne,t_sne_during_train

######################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns


######################################################################################################



def test_step(data_loader, model,model2=None):
    model = model.eval()
    if model2 is not None:
        model2 = model2.eval()
    total_num = len(data_loader.dataset.tensors[1])
    avg_accuracy = 0.
    global_step = 0
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


def valid_step(data_loader, model,model2=None):
    model = model.eval()
    if model2 is not None:
        model2 = model2.eval()
    global_step = 0
    avg_accuracy = 0.

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)

        if model2 is not None:
            logits1 = model(x)
            logits2 = model2(x)
            acc = torch.eq(torch.argmax((logits1 + logits2) / 2, 1), y)
        else:
            logits1 = model(x)
            acc = torch.eq(torch.argmax(logits1 , 1), y)
        acc = acc.cpu().numpy()
        acc = np.mean(acc)
        avg_accuracy += acc
        global_step += 1
    return avg_accuracy / global_step


def update_reduce_step(cur_step, num_gradual, tau=0.5,args=None):
    # if cur_step > args.warmup:
    return 1.0 - tau * min((cur_step) / num_gradual, 1)
    # else:
    #     return 1.0


def train_model(model, train_loader, valid_loader, test_loader, args, tau,train_dataset=None,
                noise_prior=None,noise_or_not=None,saver=None):

    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
    # learning history
    train_acc_list = []
    train_acc_list_aug = []
    train_avg_loss_list = []
    test_acc_list = []
    f1s = []
    conf_ids=[]
    try:
        loss_all = np.zeros((args.num_training_samples, args.epochs))

        max_tect_acc_among_last10 = 0
        model_state_dict = None
        conf_num = []
        for e in range(args.epochs):
            sel_dict = {'sel_ind': [], 'lam': [], 'mix_ind': []}

            # update reduce step
            rt = update_reduce_step(cur_step=e, num_gradual=args.num_gradual, tau=tau,args=args)

            # training step
            if e <= args.warmup:
                train_accuracy, avg_loss, model_new = warmup_single_model(data_loader=train_loader,
                                                                            model=model,
                                                                            optimizer=optimizer,
                                                                            criterion=criterion,
                                                                            epoch=e,
                                                                            loss_all=loss_all,
                                                                          args=args)
            else:
                if args.model in ['vanilla']:
                    train_accuracy, avg_loss, model_new = warmup_single_model(data_loader=train_loader,
                                                                               model=model,
                                                                               optimizer=optimizer,
                                                                               loss_all=loss_all,
                                                                               criterion=criterion,
                                                                               epoch=e,
                                                                               args=args)
                elif args.model in ['single_sel','single_aug_after_sel','single_ae_aug_after_sel','single_ae_sel']:
                    if args.augMSE:
                        train_accuracy, avg_loss, model_new = train_step_single_aug_after_sel_augMSE(
                            data_loader=train_loader,
                            model=model,
                            optimizer=optimizer,
                            loss_all=loss_all,
                            criterion=criterion,
                            rt=rt,
                            epoch=e,
                            args=args,
                            sel_dict=sel_dict)
                    else:
                        train_accuracy, avg_loss, model_new, confident_set_id = train_step_single_aug_after_sel(
                            data_loader=train_loader,
                            model=model,
                            optimizer=optimizer,
                            loss_all=loss_all,
                            criterion=criterion,
                            rt=rt,
                            epoch=e,
                            args=args, sel_dict=sel_dict)

                    # if e in [30, 50, 70, 100, 150, 200, 250] and args.seed == 37:
                    if args.confcsv is not None:
                        conf_num, _ = add_to_confident_set_id(args=args,
                                                           confident_set_id=confident_set_id.astype(int),
                                                           train_dataset=train_dataset, epoch=e,
                                                           conf_num=conf_num)



                elif args.model in ['single_ae_aug_before_sel']:
                    train_accuracy, avg_loss, model_new = train_step_single_aug_before_sel(data_loader=train_loader,
                                                                               model=model,
                                                                               optimizer=optimizer,
                                                                               loss_all=loss_all,
                                                                               criterion=criterion,
                                                                               rt=rt,
                                                                               epoch=e,
                                                                               args=args,
                                                                               sel_dict=sel_dict)
                elif args.model in ['single_aug','single_ae','single_ae_aug']:
                    train_accuracy, avg_loss, model_new = train_step_single_aug(data_loader=train_loader,
                                                                                         model=model,
                                                                                         optimizer=optimizer,
                                                                                         loss_all=loss_all,
                                                                                         criterion=criterion,
                                                                                         rt=rt,
                                                                                         epoch=e,
                                                                                         args=args,
                                                                                sel_dict=sel_dict)
                elif args.model in ['single_ae_aug_sel_allaug']:
                    train_accuracy, avg_loss, model_new = train_step_single_aug_sel_allaug(
                        data_loader=train_loader,
                        model=model,
                        optimizer=optimizer,
                        loss_all=loss_all,
                        criterion=criterion,
                        rt=rt,
                        epoch=e,
                        args=args,
                        sel_dict=sel_dict)

            model = model_new

            if args.tsne_during_train and args.seed == 37 and e in args.tsne_epochs:
                xs, ys, _, y_clean = train_dataset.tensors
                with torch.no_grad():
                    t_sne_during_train(xs, ys, y_clean, model=model, tsne=True, args=args,sel_dict=sel_dict,epoch=e)

            # testing/valid step
            test_accuracy, f1 = test_step(data_loader=test_loader,
                                      model=model)

            dev_accuracy = valid_step(data_loader=valid_loader,
                                      model=model)

            if args.model in ['single_ae_aug_sel_allaug','single_ae_aug_before_sel','single_ae_aug_after_sel',
                              'single_aug','single_ae_aug','single_sel','single_aug_after_sel','single_ae_sel']:
                train_acc_list.append(train_accuracy[0])
                train_acc_list_aug.append(train_accuracy[1])
                train_acc_oir =train_accuracy[0]
            else:
                train_acc_list.append(train_accuracy)
                train_acc_oir = train_accuracy
            train_avg_loss_list.append(avg_loss)
            test_acc_list.append(test_accuracy)
            f1s.append(f1)
            # conf_ids.append(confident_set_id)

            print(
                '{} epoch - Train Loss {:.4f}\tTrain accuracy {:.4f}\tDev accuracy {:.4f}\tTest accuracy {:.4f}\tReduce rate {:.4f}'.format(
                    e + 1,
                    avg_loss,
                    train_acc_oir,
                    dev_accuracy,
                    test_accuracy,
                    rt))

            if args.save_model and args.epochs > 11:
                if e >= args.epochs - 10 and max_tect_acc_among_last10 <= test_accuracy:
                    max_tect_acc_among_last10 = test_accuracy
                    model_state_dict=model.state_dict()
        if args.confcsv is not None:
            pd.DataFrame(conf_num).to_csv(os.path.join(args.basicpath,'src','bar_info',
                                                       args.dataset+str(args.auto_rate)+args.confcsv),mode='a',header=True)

        if model_state_dict is not None: # save model
            model_to_save_dir = os.path.join(args.basicpath, 'src', 'model_save', args.dataset)
            os.makedirs(model_to_save_dir,exist_ok=True)
            datestr=time.strftime(('%Y%m%d'))
            if args.label_noise == -1:
                label_noise = 'inst{}'.format(int(args.ni[0] * 100))
            elif args.label_noise == 0:
                label_noise = 'sym{}'.format(int(args.ni[0] * 100))
            else:
                label_noise = 'asym{}'.format(int(args.ni[0] * 100))
            filename=os.path.join(model_to_save_dir, args.model)
            filename_sel_dict ='{}{}_{}_{}_sel_dict.npy'.format(filename, args.aug, label_noise, datestr)
            np.save(filename_sel_dict,sel_dict) # save sel_ind
            filename='{}{}_{}_{}.pt'.format(filename, args.aug, label_noise, datestr)
            torch.save(model_state_dict, filename) # save model

    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    if args.plt_loss_hist:
        plot_train_loss_and_test_acc(train_avg_loss_list,test_acc_list,args,pred_precision=train_acc_list,aug_accs=train_acc_list_aug,
                                     saver=saver,save=True)

    if args.plot_tsne and model_state_dict is not None and args.seed==37:

        xs,ys,_,y_clean = train_dataset.tensors
        classifier1 = NonLinClassifier(args.embedding_size, args.nbins, d_hidd=args.classifier_dim, dropout=args.dropout,
                                       norm=args.normalization)

        model = CNNAE(input_size=xs.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                      seq_len=xs.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                      padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)
        if args.model in ['single_ae_aug_after_sel', 'single_ae', 'single_ae_sel', 'single_ae_aug',
                          'single_ae_aug_before_sel','single_ae_aug_sel_allaug']:
            model = MetaModel_AE(ae=model, classifier=classifier1, name='CNN').to(device)
        elif args.model in ['single_aug_after_sel', 'single_aug', 'single_sel', 'vanilla']:
            model = MetaModel(ae=model, classifier=classifier1, name='CNN').to(device)

        with torch.no_grad():

            t_sne(xs, ys, y_clean,model=model, tsne=True, args=args,datestr=datestr,sel_dict=sel_dict)

    training_results = dict()
    training_results['max_valid_acc'] = np.max(test_acc_list)
    training_results['max_valid_acc_epoch'] = np.argmax(test_acc_list)
    training_results['avg_last_ten_valid_acc'] = np.mean(test_acc_list[-10:])
    training_results['max_valid_f1'] = np.max(f1s)
    training_results['max_valid_f1_epoch'] = np.argmax(f1s)
    training_results['avg_last_ten_valid_f1'] = np.mean(f1s[-10:])
    return model, training_results


def train_eval_model(model, x_train, x_valid, x_test, Y_train, Y_valid, Y_test, Y_train_clean, Y_valid_clean,
                     ni, args, saver, plt_embedding=True, plt_cm=True,mask_train=None):

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))), torch.from_numpy(Y_train_clean))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(Y_valid).long())
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                   num_workers=args.num_workers)

    # compute noise prior
    noise_prior = compute_noise_prior(Y_train,args)
    noise_or_not = mask_train
    ######################################################################################################
    # Train model

    model, training_results = train_model(model, train_loader, valid_loader, test_loader, args, ni,
                                          train_dataset=train_dataset,noise_prior=noise_prior,noise_or_not=noise_or_not,
                                          saver=saver)
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


def main_wrapper_single_model(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean, saver,seeds=[0]):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)
            self.args=args
            self.path = path
            self.makedir_()
            # self.make_log()

    classes = len(np.unique(Y_train_clean))
    args.nbins = classes
    history = x_train.shape[1]

    # Network definition
    classifier1 = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                   norm=args.normalization)

    model = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                   seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                   padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)


    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    if args.model in ['single_ae_aug_after_sel','single_ae','single_ae_sel','single_ae_aug','single_ae_aug_before_sel',
                      'single_ae_aug_sel_allaug']:
        model = MetaModel_AE(ae=model, classifier=classifier1, name='CNN').to(device)
    elif args.model in ['single_aug_after_sel','single_aug','single_sel','vanilla']:
        model = MetaModel(ae=model, classifier=classifier1, name='CNN').to(device)


    nParams = sum([p.nelement() for p in model.parameters()])
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

        saver_loop = SaverSlave(os.path.join(saver.path, f'seed_{seed}'))
        # saver_loop.append_str(['SEED: {}'.format(seed), '\r\n'])

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
            model = reset_model(model)

            Y_train, mask_train = flip_label(x_train,Y_train_clean, ni, args)
            Y_valid, mask_valid = flip_label(x_valid,Y_valid_clean, ni, args)
            Y_test = Y_test_clean

            train_results, valid_results, test_results = train_eval_model(model, x_train, x_valid, x_test, Y_train,
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
            df_results = df_results.append(test_results, ignore_index=True)

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


def warmup_single_model(data_loader, model, optimizer, criterion,epoch=None,
                         loss_all=None,args=None):
    global_step = 0
    avg_accuracy = 0.
    avg_loss1 = 0.
    # if_get_feature = True
    model = model.train()

    for batch_idx,(x, y_hat,x_idx,mask_train) in enumerate(data_loader):
        if x.shape[0]==1:
            continue
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)
        y1 = y_hat

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

        avg_loss1 += model_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out, 1), y1).float()
        avg_accuracy += acc.mean().cpu().numpy()
        global_step += 1



    avg_loss=avg_loss1
    if args.model in ['single_ae_aug_before_sel','single_ae_aug_sel_allaug','single_ae_aug_after_sel',
                      'single_aug','single_ae_aug','single_sel','single_aug_after_sel','single_ae_sel']:
        try:
            return (avg_accuracy / global_step,0.), avg_loss / global_step, model
        except:
            print('\n dataloader:\n',data_loader)
            raise
    else:
        try:
            return avg_accuracy / global_step, avg_loss / global_step, model
        except:
            print('\n dataloader:\n',data_loader)
            raise



def train_step_single_aug_after_sel(data_loader, model, optimizer,  criterion, rt,fit=None,loss_all=None,
               p_threshold=None, normalization=None,epoch=0,args=None,sel_dict=None,estimate_noise_rate=None):
    global_step = 0
    aug_step=0
    avg_accuracy = 0.
    avg_accuracy_aug = 0.
    avg_loss = 0.
    model = model.train()
    confident_set_id=np.array([])


    for batch_idx,(x, y_hat,x_idx,y_clean) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat,y_clean = x.to(device), y_hat.to(device),y_clean.to(device)
        y = y_hat

        h = model.encoder(x)
        out = model.classifier(h.squeeze(-1))

        if hasattr(model,'decoder'):
            hd = model.decoder(h)
            recon_loss=nn.MSELoss(reduction='mean')(hd, x)
        else:
            recon_loss=0.

        model_loss = criterion(out, y_hat)

        if loss_all is not None:
            loss_all[x_idx,epoch]=model_loss.data.detach().clone().cpu().numpy()

        if args.auto_rate in [1,3]:
            model_loss, model_sel_idx = small_loss_criterion_auto_rate1(loss_all=loss_all, labels=y_hat, p_threshold=0.5,
                                                                   model_loss=model_loss,args=args,
                                                                   epoch=epoch, x_idxs=x_idx)
        elif args.auto_rate == 2:
            model_loss, model_sel_idx = small_loss_criterion_auto_rate2(model_loss=model_loss,
                                                                        loss_all=loss_all, args=args,
                                                                        epoch=epoch, x_idxs=x_idx, labels=y_hat,
                                                                        y_clean=y_clean)
        elif args.auto_rate == 4:
            model_loss, model_sel_idx = small_loss_criterion_without_eliminate(model_loss=model_loss, rt=rt,
                                                        loss_all=loss_all, args=args,
                                                        epoch=epoch,x_idxs=x_idx)
        else:
            model_loss,model_sel_idx = small_loss_criterion(model_loss=model_loss, rt=rt,
                                                        loss_all=loss_all, args=args,
                                                        epoch=epoch,x_idxs=x_idx)




        # data Augmentation after selecting clean samples
        if args.aug == 'NoAug':
            aug_model_loss = 0.
            aug_recon_loss = 0.
            avg_accuracy_aug = 0.
        elif (batch_idx % args.arg_interval == 0) and len(model_sel_idx)!=1:

            if args.aug=='GNoise':
                x_aug = torch.from_numpy(
                    tsaug.AddNoise(scale=0.015).augment(x[model_sel_idx].cpu().numpy())).float().to(device)
            elif args.aug=='Oversample':
                x_aug = x[model_sel_idx].detach().clone()
            elif args.aug=='Convolve':
                x_aug = torch.from_numpy(
                    tsaug.Convolve(window='flattop',size=10).augment(x[model_sel_idx].cpu().numpy())).float().to(device)
            elif args.aug=='Crop':
                x_aug = torch.from_numpy(
                    tsaug.Crop(size=int(args.sample_len*(2/3)),resize=int(args.sample_len)).augment(x[model_sel_idx].cpu().numpy())).float().to(
                    device)
            elif args.aug=='Drift':
                x_aug = torch.from_numpy(
                    tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(
                        x[model_sel_idx].cpu().numpy())).float().to(
                    device)
            elif args.aug=='TimeWarp':
                x_aug = torch.from_numpy(
                    tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                        x[model_sel_idx].cpu().numpy())).float().to(
                    device)
            elif args.aug=='Mixup':
                mix_alpha = 0.2
                lam = np.random.beta(mix_alpha, mix_alpha)
                index = torch.randperm(model_sel_idx.size(0)).to(device)
                aug_h = lam * h[model_sel_idx] + (1 - lam) * h[model_sel_idx][index, :]
                ta, tb = y_hat[model_sel_idx], y_hat[model_sel_idx][index]
                outx_aug = model.classifier(aug_h.squeeze(-1))
                y_clean_aug =torch.argmax(torch.from_numpy(lam*to_one_hot(args.nbins,y_clean[model_sel_idx].cpu().numpy())+
                               (1-lam)*to_one_hot(args.nbins,y_clean[model_sel_idx][index].cpu().numpy())),dim=1).to(device)
                model_loss_mix = lam * criterion(outx_aug, ta) + (1 - lam) * criterion(outx_aug, tb)
            else:
                pass

            if args.aug=='Mixup':
                aug_model_loss=model_loss_mix.sum()
                if epoch==args.epochs-1 or epoch in args.tsne_epochs:
                    sel_dict['lam'].append(lam)
                    sel_dict['mix_ind'].append(x_idx[model_sel_idx][index].cpu().numpy())
                if args.aug_ae:
                    if hasattr(model, 'decoder'):
                        aug_hd = model.decoder(aug_h)
                    aug_recon_loss = nn.MSELoss(reduction='mean')(aug_hd, lam * x.clone()[model_sel_idx] + (1 - lam) * x.clone()[model_sel_idx][index, :])
                else:
                    aug_recon_loss = 0.
            else:
                if len(x_aug)==1:
                    aug_model_loss = 0.
                    aug_recon_loss = 0.
                else:
                    aug_h = model.encoder(x_aug)
                    if args.aug_ae:
                        if hasattr(model, 'decoder'):
                            aug_hd = model.decoder(aug_h)
                        aug_recon_loss = nn.MSELoss(reduction='mean')(aug_hd, x_aug)
                    else:
                        aug_recon_loss = 0.
                    outx_aug=model.classifier(aug_h.squeeze(-1))
                    y_clean_aug = y_clean[model_sel_idx]
                    aug_model_loss = criterion(outx_aug, y_hat[model_sel_idx]).sum()

            if epoch==args.epochs-1 or epoch in args.tsne_epochs:
                sel_dict['sel_ind'].append(x_idx[model_sel_idx].cpu().numpy())


            avg_accuracy_aug += torch.eq(torch.argmax(outx_aug, 1), y_clean_aug).float().mean().cpu().numpy()
            aug_step+=1



        else:
            aug_model_loss = 0.
            aug_recon_loss=0.
            avg_accuracy_aug = 0.

        model_loss = model_loss + args.L_aug_coef*aug_model_loss+args.L_rec_coef*(recon_loss+aug_recon_loss)

        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc1 = torch.eq(torch.argmax(out[model_sel_idx], 1), y[model_sel_idx]).float()
        avg_accuracy += acc1.mean().cpu().numpy()

        global_step += 1

        confident_set_id=np.concatenate((confident_set_id,x_idx[model_sel_idx].cpu().numpy()))

    if args.aug == 'NoAug':
        return (avg_accuracy / global_step, 0.), avg_loss / global_step, model,confident_set_id
    else:
        if aug_step == 0:
             aug_step = 1.
        return (avg_accuracy / global_step, avg_accuracy_aug / aug_step), avg_loss / global_step, model,confident_set_id

def train_step_single_aug(data_loader, model, optimizer, criterion,
                                    loss_all=None,rt=None,fit=None,p_threshold=None, normalization=None,
                                       epoch=0,args=None,sel_dict=None):
    global_step = 0
    aug_step = 0
    avg_accuracy = 0.
    avg_accuracy_aug = 0.
    avg_loss = 0.

    model = model.train()

    for batch_idx,(x, y_hat,x_idx,y_clean) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat,y_clean = x.to(device), y_hat.to(device),y_clean.to(device)
        y = y_hat

        h = model.encoder(x)
        out = model.classifier(h.squeeze(-1))
        if hasattr(model,'decoder'):
            hd = model.decoder(h)
            recon_loss=nn.MSELoss(reduction='mean')(hd, x)
        else:
            recon_loss=0.

        model_loss = criterion(out, y_hat)

        if args.aug == 'NoAug':
            aug_model_loss = 0.
        elif batch_idx % args.arg_interval == 0:

            if args.aug == 'GNoise':
                x_aug = torch.from_numpy(
                    tsaug.AddNoise(scale=0.015).augment(x.cpu().numpy())).float().to(device)
            elif args.aug == 'Oversample':
                x_aug = x.detach().clone()
            elif args.aug == 'Convolve':
                x_aug = torch.from_numpy(
                    tsaug.Convolve(window='flattop', size=10).augment(x.cpu().numpy())).float().to(device)
            elif args.aug == 'Crop':
                x_aug = torch.from_numpy(
                    tsaug.Crop(size=int(args.sample_len * (2 / 3)), resize=int(args.sample_len)).augment(
                        x.cpu().numpy())).float().to(
                    device)
            elif args.aug == 'Drift':
                x_aug = torch.from_numpy(
                    tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(
                        x.cpu().numpy())).float().to(
                    device)
            elif args.aug == 'TimeWarp':
                x_aug = torch.from_numpy(
                    tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                        x.cpu().numpy())).float().to(
                    device)
            elif args.aug == 'Mixup':
                mix_alpha = 0.2
                lam = np.random.beta(mix_alpha, mix_alpha)
                index = torch.randperm(x_idx.size(0)).to(device)
                mix_h = lam * h + (1 - lam) * h[index, :]
                ta, tb = y_hat, y_hat[index]
                outx_aug = model.classifier(mix_h.squeeze(-1))
                y_clean_aug = torch.argmax(torch.from_numpy(lam * to_one_hot(args.nbins, y_clean.cpu().numpy()) +
                                                            (1 - lam) * to_one_hot(args.nbins,
                                                                                   y_clean[index].cpu().numpy())),
                                           dim=1).to(device)

                model_loss_mix = lam * criterion(outx_aug, ta) + (1 - lam) * criterion(outx_aug, tb)

            else:
                pass

            if args.aug == 'Mixup':
                aug_model_loss = model_loss_mix.sum()
                if epoch == args.epochs - 1:
                    sel_dict['lam'].append(lam)
                    sel_dict['mix_ind'].append(x_idx[index].cpu().numpy())
            else:
                outx_aug = model(x_aug)
                y_clean_aug = y_clean

                aug_model_loss = criterion(outx_aug, y_hat).sum()

            if epoch==args.epochs-1:
                sel_dict['sel_ind'].append(x_idx.cpu().numpy())
            avg_accuracy_aug += torch.eq(torch.argmax(outx_aug, 1), y_clean_aug).float().mean().cpu().numpy()
            aug_step += 1
        else:
            aug_model_loss = 0.


        model_loss = model_loss.sum() + aug_model_loss+recon_loss

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

    if args.aug == 'NoAug':
        return (avg_accuracy / global_step, 0.), avg_loss / global_step, model
    else:
        return (avg_accuracy / global_step, avg_accuracy_aug / aug_step), avg_loss / global_step, model


def train_step_single_aug_before_sel(data_loader, model, optimizer,  criterion, rt,fit=None,loss_all=None,
               p_threshold=None, normalization=None,epoch=0,args=None,sel_dict=None):
    global_step = 0
    aug_step=0
    avg_accuracy = 0.
    avg_accuracy_aug = 0.
    avg_loss = 0.
    model = model.train()

    for batch_idx,(x, y_hat,x_idx,y_clean) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat,y_clean = x.to(device), y_hat.to(device),y_clean.to(device)
        y = y_hat

        h = model.encoder(x)
        out = model.classifier(h.squeeze(-1))

        if hasattr(model,'decoder'):
            hd = model.decoder(h)
            recon_loss=nn.MSELoss(reduction='mean')(hd, x)
        else:
            recon_loss=0.

        model_loss = criterion(out, y_hat)

        if loss_all is not None:
            loss_all[x_idx,epoch]=model_loss.data.detach().clone().cpu().numpy()

        if args.auto_rate == 1:
            model_loss, sel_index = small_loss_criterion_auto_rate1(model_loss=model_loss,
                                                                        loss_all=loss_all, args=args,
                                                                        epoch=epoch, x_idxs=x_idx, labels=y_hat)
        elif args.auto_rate == 2:
            model_loss, sel_index = small_loss_criterion_auto_rate2(model_loss=model_loss,
                                                                        loss_all=loss_all, args=args,
                                                                        epoch=epoch, x_idxs=x_idx, labels=y_hat)
        else:
            model_loss, sel_index = small_loss_criterion(model_loss=model_loss, rt=rt,
                                                             loss_all=loss_all, args=args,
                                                             epoch=epoch, x_idxs=x_idx)


        # data Augmentation after selecting clean samples
        if args.aug == 'NoAug':
            aug_model_loss = 0.
        elif batch_idx % args.arg_interval == 0:

            if args.aug=='GNoise':
                x_aug = torch.from_numpy(
                    tsaug.AddNoise(scale=0.015).augment(x.cpu().numpy())).float().to(device)
                # x_aug = torch.from_numpy(
                #     tsaug.AddNoise(scale=0.15).augment(x.cpu().numpy())).float().to(device)
            elif args.aug=='Oversample':
                x_aug = x.detach().clone()
            elif args.aug=='Convolve':
                x_aug = torch.from_numpy(
                    tsaug.Convolve(window='flattop',size=10).augment(x.cpu().numpy())).float().to(device)
            elif args.aug=='Crop':
                x_aug = torch.from_numpy(
                    tsaug.Crop(size=int(args.sample_len*(2/3)),resize=int(args.sample_len)).augment(x.cpu().numpy())).float().to(
                    device)
            elif args.aug=='Drift':
                x_aug = torch.from_numpy(
                    tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(
                        x.cpu().numpy())).float().to(
                    device)
            elif args.aug=='TimeWarp':
                x_aug = torch.from_numpy(
                    tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                        x.cpu().numpy())).float().to(
                    device)
            elif args.aug=='Mixup':
                mix_alpha = 0.2
                lam = np.random.beta(mix_alpha, mix_alpha)
                index = torch.randperm(x_idx.size(0)).to(device)
                mix_h = lam * h + (1 - lam) * h[index, :]
                ta, tb = y_hat, y_hat[index]
                outx_aug = model.classifier(mix_h.squeeze(-1))
                y_clean_aug = torch.argmax(torch.from_numpy(lam * to_one_hot(args.nbins, y_clean.cpu().numpy()) +
                                        (1 - lam) * to_one_hot(args.nbins, y_clean[index].cpu().numpy())),
                                        dim=1).to(device)

                model_loss_mix = lam * criterion(outx_aug, ta) + (1 - lam) * criterion(outx_aug, tb)
            else:
                pass

            if args.aug=='Mixup':
                aug_model_loss=model_loss_mix
                if epoch==args.epochs-1 or epoch in args.tsne_epochs:
                    sel_dict['lam'].append(lam)
                    sel_dict['mix_ind'].append(x_idx[index].cpu().numpy())
            else:
                outx_aug = model(x_aug)
                y_clean_aug=y_clean

                aug_model_loss = criterion(outx_aug, y_hat)
            aug_model_loss, aug_sel_index = small_loss_criterion(model_loss=aug_model_loss, rt=rt, args=args,
                                                 epoch=epoch, x_idxs=x_idx)
            if epoch==args.epochs-1 or epoch in args.tsne_epochs:
                sel_dict['sel_ind'].append(x_idx.cpu().numpy())
            avg_accuracy_aug += torch.eq(torch.argmax(outx_aug[aug_sel_index], 1), y_clean_aug[aug_sel_index]).float().mean().cpu().numpy()
            aug_step += 1

        else:
            aug_model_loss = 0.

        model_loss = model_loss + aug_model_loss+recon_loss

        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc = torch.eq(torch.argmax(out[sel_index], 1), y[sel_index]).float()
        avg_accuracy += acc.mean().cpu().numpy()
        global_step += 1
    if args.aug=='NoAug':
        return (avg_accuracy / global_step, 0.), avg_loss / global_step, model
    else:
        return (avg_accuracy / global_step, avg_accuracy_aug / aug_step), avg_loss / global_step, model

def train_step_single_aug_after_sel_augMSE(data_loader, model, optimizer,  criterion, rt,fit=None,loss_all=None,
               p_threshold=None, normalization=None,epoch=0,args=None,sel_dict=None):
    global_step = 0
    aug_step=0
    avg_accuracy = 0.
    avg_accuracy_aug = 0.
    avg_loss = 0.
    if_get_feature = True
    model = model.train()

    for batch_idx,(x, y_hat,x_idx,y_clean) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat,y_clean = x.to(device), y_hat.to(device),y_clean.to(device)
        y = y_hat

        h = model.encoder(x)
        out = model.classifier(h.squeeze(-1))

        if hasattr(model,'decoder'):
            hd = model.decoder(h)
            recon_loss=nn.MSELoss(reduction='mean')(hd, x)
        else:
            recon_loss=0.

        model_loss = criterion(out, y_hat)

        if loss_all is not None:
            loss_all[x_idx,epoch]=model_loss.data.detach().clone().cpu().numpy()

        if args.auto_rate == 1:
            model_loss, model_sel_idx = small_loss_criterion_auto_rate1(model_loss=model_loss,
                                                                        loss_all=loss_all, args=args,
                                                                        epoch=epoch, x_idxs=x_idx, labels=y_hat)
        elif args.auto_rate == 2:
            model_loss, model_sel_idx = small_loss_criterion_auto_rate2(model_loss=model_loss,
                                                                        loss_all=loss_all, args=args,
                                                                        epoch=epoch, x_idxs=x_idx, labels=y_hat,y_clean=y_clean)
        else:
            model_loss,model_sel_idx = small_loss_criterion(model_loss=model_loss, rt=rt,
                                                        loss_all=loss_all, args=args,
                                                        epoch=epoch,x_idxs=x_idx)

        # data Augmentation after selecting clean samples
        if args.aug == 'NoAug':
            aug_model_loss = 0.
        elif (batch_idx % args.arg_interval == 0) and len(model_sel_idx)>0:

            if args.aug=='GNoise':
                x_aug = torch.from_numpy(
                    tsaug.AddNoise(scale=0.015).augment(x[model_sel_idx].cpu().numpy())).float().to(device)
            elif args.aug=='Oversample':
                x_aug = x[model_sel_idx].detach().clone()
            elif args.aug=='Convolve':
                x_aug = torch.from_numpy(
                    tsaug.Convolve(window='flattop',size=10).augment(x[model_sel_idx].cpu().numpy())).float().to(device)
            elif args.aug=='Crop':
                x_aug = torch.from_numpy(
                    tsaug.Crop(size=int(args.sample_len*(2/3)),resize=int(args.sample_len)).augment(x[model_sel_idx].cpu().numpy())).float().to(
                    device)
            elif args.aug=='Drift':
                x_aug = torch.from_numpy(
                    tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(
                        x[model_sel_idx].cpu().numpy())).float().to(
                    device)
            elif args.aug=='TimeWarp':
                x_aug = torch.from_numpy(
                    tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                        x[model_sel_idx].cpu().numpy())).float().to(
                    device)
            elif args.aug=='Mixup':
                mix_alpha = 0.2
                lam = np.random.beta(mix_alpha, mix_alpha)
                index = torch.randperm(model_sel_idx.size(0)).to(device)
                mix_h = lam * h[model_sel_idx] + (1 - lam) * h[model_sel_idx][index, :]
                with torch.no_grad():
                    ta, tb = (torch.softmax(out).detch())[model_sel_idx], (torch.softmax(out).detch())[model_sel_idx][index]
                outx_aug = model.classifier(mix_h.squeeze(-1))
                y_clean_aug =torch.argmax(torch.from_numpy(lam*to_one_hot(args.nbins,y_clean[model_sel_idx].cpu().numpy())+
                               (1-lam)*to_one_hot(args.nbins,y_clean[model_sel_idx][index].cpu().numpy())),dim=1).to(device)
                model_loss_mix = lam * ((outx_aug- ta)**2) + (1 - lam) * ((outx_aug- tb)**2)
            else:
                pass

            if args.aug=='Mixup':
                aug_model_loss=model_loss_mix.mean()
                if epoch==args.epochs-1 or epoch in args.tsne_epochs:
                    sel_dict['lam'].append(lam)
                    sel_dict['mix_ind'].append(x_idx[model_sel_idx][index].cpu().numpy())
            else:
                if len(x_aug)==1:
                    aug_model_loss = 0.
                else:
                    outx_aug = model(x_aug)
                    y_clean_aug = y_clean[model_sel_idx]
                    probs_u = torch.softmax(outx_aug, dim=1)
                    with torch.no_grad():
                        targets_u = torch.softmax(out.detach(), dim=1)
                    aug_model_loss = torch.mean((probs_u - targets_u[model_sel_idx]) ** 2)


            if (epoch==args.epochs-1 or epoch in args.tsne_epochs) and len(x_aug)!=1:
                sel_dict['sel_ind'].append(x_idx[model_sel_idx].cpu().numpy())

            if len(x_aug)!=1:
                avg_accuracy_aug += torch.eq(torch.argmax(outx_aug, 1), y_clean_aug).float().mean().cpu().numpy()
                aug_step+=1
            else:
                avg_accuracy_aug = 0.
        else:
            aug_model_loss = 0.

        model_loss = model_loss + aug_model_loss+recon_loss

        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc1 = torch.eq(torch.argmax(out[model_sel_idx], 1), y[model_sel_idx]).float()
        avg_accuracy += acc1.mean().cpu().numpy()

        global_step += 1

    if args.aug == 'NoAug':
        return (avg_accuracy / global_step, 0.), avg_loss / global_step, model
    else:
        if aug_step == 0:
             aug_step = 1.
        return (avg_accuracy / global_step, avg_accuracy_aug / aug_step), avg_loss / global_step, model

def train_step_single_aug_sel_allaug(data_loader, model, optimizer,  criterion, rt,fit=None,loss_all=None,
               p_threshold=None, normalization=None,epoch=0,args=None,sel_dict=None):
    global_step = 0
    aug_step=0
    avg_accuracy = 0.
    avg_accuracy_aug = 0.
    avg_loss = 0.
    model = model.train()

    for batch_idx,(x, y_hat,x_idx,y_clean) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat,y_clean = x.to(device), y_hat.to(device),y_clean.to(device)
        y = y_hat

        h = model.encoder(x)
        out = model.classifier(h.squeeze(-1))

        if hasattr(model,'decoder'):
            hd = model.decoder(h)
            recon_loss=nn.MSELoss(reduction='mean')(hd, x)
        else:
            recon_loss=0.

        model_loss = criterion(out, y_hat)

        if loss_all is not None:
            loss_all[x_idx,epoch]=model_loss.data.detach().clone().cpu().numpy()

        if args.auto_rate == 1:
            model_loss, model_sel_idx = small_loss_criterion_auto_rate1(model_loss=model_loss,
                                                                        loss_all=loss_all, args=args,
                                                                        epoch=epoch, x_idxs=x_idx, labels=y_hat)
        elif args.auto_rate == 2:
            model_loss, model_sel_idx = small_loss_criterion_auto_rate2(model_loss=model_loss,
                                                                        loss_all=loss_all, args=args,
                                                                        epoch=epoch, x_idxs=x_idx, labels=y_hat,y_clean=y_clean)
        else:
            model_loss,model_sel_idx = small_loss_criterion(model_loss=model_loss, rt=rt,
                                                        loss_all=loss_all, args=args,
                                                        epoch=epoch,x_idxs=x_idx)

        if args.aug == 'NoAug':
            aug_model_loss = 0.
        elif batch_idx % args.arg_interval == 0:

            if args.aug=='GNoise':
                x_aug = torch.from_numpy(
                    tsaug.AddNoise(scale=0.015).augment(x.cpu().numpy())).float().to(device)
            elif args.aug=='Oversample':
                x_aug = x.detach().clone()
            elif args.aug=='Convolve':
                x_aug = torch.from_numpy(
                    tsaug.Convolve(window='flattop',size=10).augment(x.cpu().numpy())).float().to(device)
            elif args.aug=='Crop':
                x_aug = torch.from_numpy(
                    tsaug.Crop(size=int(args.sample_len*(2/3)),resize=int(args.sample_len)).augment(x.cpu().numpy())).float().to(
                    device)
            elif args.aug=='Drift':
                x_aug = torch.from_numpy(
                    tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(
                        x.cpu().numpy())).float().to(
                    device)
            elif args.aug=='TimeWarp':
                x_aug = torch.from_numpy(
                    tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                        x.cpu().numpy())).float().to(
                    device)
            elif args.aug=='Mixup':
                mix_alpha = 0.2
                lam = np.random.beta(mix_alpha, mix_alpha)
                index = torch.randperm(x_idx.size(0)).to(device)
                mix_h = lam * h + (1 - lam) * h[index, :]
                with torch.no_grad():
                    ta, tb = (torch.softmax(out).detch()), (torch.softmax(out).detch())[index]
                outx_aug = model.classifier(mix_h.squeeze(-1))
                y_clean_aug =torch.argmax(torch.from_numpy(lam*to_one_hot(args.nbins,y_clean.cpu().numpy())+
                               (1-lam)*to_one_hot(args.nbins,y_clean[index].cpu().numpy())),dim=1).to(device)
                model_loss_mix = lam * ((outx_aug - ta) ** 2) + (1 - lam) * ((outx_aug - tb) ** 2)
            else:
                pass

            if args.aug=='Mixup':
                aug_model_loss=model_loss_mix.sum()
                if epoch==args.epochs-1 or epoch in args.tsne_epochs:
                    sel_dict['lam'].append(lam)
                    sel_dict['mix_ind'].append(x_idx[index].cpu().numpy())
            else:
                if len(x_aug)==1:
                    aug_model_loss = 0.
                else:
                    outx_aug = model(x_aug)
                    y_clean_aug = y_clean
                    probs_u = torch.softmax(outx_aug, dim=1)
                    with torch.no_grad():
                        targets_u = torch.softmax(out.detach(), dim=1)
                    aug_model_loss = torch.sum((probs_u - targets_u) ** 2)

            if (epoch==args.epochs-1 or epoch in args.tsne_epochs) and len(x_aug)!=1:
                sel_dict['sel_ind'].append(x_idx[model_sel_idx].cpu().numpy())

            if len(x_aug)!=1:
                avg_accuracy_aug += torch.eq(torch.argmax(outx_aug, 1), y_clean_aug).float().mean().cpu().numpy()
                aug_step+=1
            else:
                avg_accuracy_aug = 0.
        else:
            aug_model_loss = 0.

        model_loss = model_loss + aug_model_loss+recon_loss

        # loss exchange
        optimizer.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        acc1 = torch.eq(torch.argmax(out[model_sel_idx], 1), y[model_sel_idx]).float()
        avg_accuracy += acc1.mean().cpu().numpy()

        global_step += 1

    if args.aug == 'NoAug':
        return (avg_accuracy / global_step, 0.), avg_loss / global_step, model
    else:
        if aug_step == 0:
             aug_step = 1.
        return (avg_accuracy / global_step, avg_accuracy_aug / aug_step), avg_loss / global_step, model

def add_to_confident_set_id(args=None,confident_set_id=None,train_dataset=None,epoch=None,conf_num=None):

    xs, ys, _, y_clean = train_dataset.tensors
    ys=ys.cpu().numpy()
    y_clean=y_clean.cpu().numpy()
    TP_all=0
    FP_all=0
    for i in range(args.nbins):
        confnum_row = dict()
        confnum_row['epoch'] = epoch
        if args.auto_rate == 2:
            confnum_row['method']='Our model'
        else:
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

