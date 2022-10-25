from __future__ import print_function

import copy

# from dataset.cifar_dataset import *

import torch.utils.data as data
import sys

sys.path.append('../Sel-CL_utils')
# from models.preact_resnet import *

# from dataset.cifar_dataset import *

from torch import optim
import random
import sys
import pandas as pd
import shutil


sys.path.append('../Sel-CL_utils')
from src.utils.Sel_CL_utils.utils_noise import *
from src.utils.Sel_CL_utils.test_eval import test_eval
from src.utils.Sel_CL_utils.queue_with_pro import *
from src.utils.Sel_CL_utils.kNN_test import kNN
from src.utils.Sel_CL_utils.MemoryMoCo import MemoryMoCo
from src.utils.Sel_CL_utils.other_utils import *
# from src.models.preact_resnet import *
from src.utils.Sel_CL_utils.lr_scheduler import get_scheduler
from apex import amp
from src.models.MultiTaskClassification import MetaModel_Sel_CL

from torch.utils.data import DataLoader, TensorDataset

from src.models.MultiTaskClassification import NonLinClassifier
from src.models.model import CNNAE
from src.utils.saver import Saver
from src.utils.utils import readable, reset_seed_, reset_model, flip_label, map_abg, remove_empty_dirs, \
    evaluate_class

#####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns
#####

def build_model(args, device):
    model = PreActResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)
    model_ema = PreActResNet18(num_classes=args.num_classes, low_dim=args.low_dim, head=args.headType).to(device)

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)
    return model, model_ema


def main_wrapper_Sel_CL(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean, saver,seeds=[0]):
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
    classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                   norm=args.normalization)

    model = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                   seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                   padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

    classifier_ema = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                  norm=args.normalization)

    model_ema = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                  seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                  padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

    ######################################################################################################
    # model is multi task - AE Branch and Classification branch
    model = MetaModel_Sel_CL(ae=model, classifier=classifier, name='CNN', low_dim=args.low_dim).to(device)
    model_ema = MetaModel_Sel_CL(ae=model_ema, classifier=classifier_ema, name='CNN', low_dim=args.low_dim).to(device)

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

        reset_seed_(seed)
        model = reset_model(model)
        model_ema = copy.deepcopy(model)
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
            model = reset_model(model)

            Y_train, mask_train = flip_label(x_train, Y_train_clean, ni, args)
            Y_valid, mask_valid = flip_label(x_valid, Y_valid_clean, ni, args)
            Y_test = Y_test_clean

            ############################################################

            exp_path = os.path.join(args.out,
                                    'noise_models_FCN_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                              args.seed_initialization,
                                                                              args.seed_dataset),
                                    args.noise_type, str(int(args.ni[0]*100)))
            res_path = os.path.join(args.out, 'metrics_FCN_{0}_SI{1}_SD{2}'.format(args.experiment_name,
                                                                                   args.seed_initialization,
                                                                                   args.seed_dataset),
                                    args.noise_type, str(int(args.ni[0]*100)))

            if not os.path.isdir(res_path):
                os.makedirs(res_path)

            if not os.path.isdir(exp_path):
                os.makedirs(exp_path)

            __console__ = sys.stdout
            name = "/results"
            # log_file = open(res_path + name + ".log", 'a')
            # sys.stdout = log_file
            print(args)

            random.seed(args.seed_initialization)  # python seed for image transformation

            ############################################################

            valid_results, test_results = train_eval_model(model, x_train, x_valid, x_test, Y_train,
                                                                          Y_valid, Y_test, Y_train_clean,
                                                                          Y_valid_clean,
                                                                          ni, args, saver_slave,
                                                                          mask_train=mask_train,
                                                                          res_path=res_path,
                                                                          exp_path=exp_path,
                                                                          model_ema=model_ema)

            keys = list(test_results.keys())
            test_results['noise'] = ni
            test_results['seed'] = seed
            test_results['correct'] = 'Co-teaching'
            test_results['losses'] = map_abg([0, 1, 0])
            # saver_loop.append_str(['Test Results:'])
            # saver_loop.append_dict(test_results)
            # df_results = df_results.append(test_results, ignore_index=True)

    remove_empty_dirs(saver.path)
    return test_results

def train_eval_model(model, x_train, x_valid, x_test, Y_train, Y_valid, Y_test, Y_train_clean, Y_valid_clean,
                     ni, args, saver, plt_embedding=False, plt_cm=False,mask_train=None,exp_path=None,
                     res_path=None, model_ema = None, log_file=None):

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))),torch.from_numpy(Y_train_clean))
    valid_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(Y_valid).long())
    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                   num_workers=args.num_workers)

    ######################################################################################################
    uns_contrast = MemoryMoCo(args.low_dim, args.uns_queue_k, args.uns_t, thresh=0).cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", num_losses=2)
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    if args.sup_queue_use == 1:
        queue = queue_with_pro(args, device)
    else:
        queue = []

    # np.save(res_path + '/' + str(int(args.ni[0]*100)) + '_noisy_labels.npy', np.asarray(train_dataset.tensors[1]))
    
    accs=[]
    f1s=[]
    for epoch in range(args.initial_epoch, args.epoch + 1):

        st = time.time()
        print("=================>    ", args.experiment_name, args.ni[0])
        if (epoch <= args.warmup):
            if (args.warmup_way == 'uns'):
                train_uns(args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader,
                          optimizer,
                          epoch)
            else:
                train_selected_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                                    num_workers=args.num_workers,
                                                                    pin_memory=True,
                                                                    sampler=torch.utils.data.WeightedRandomSampler(
                                                                        torch.ones(len(train_dataset)),
                                                                        len(train_dataset)))
                trainNoisyLabels = torch.LongTensor(train_loader.dataset.targets).unsqueeze(1)
                train_sup(args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader,
                          train_selected_loader, optimizer, epoch,
                          torch.eq(trainNoisyLabels, trainNoisyLabels.t()))
        else:
            train_selected_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                                num_workers=args.num_workers,
                                                                pin_memory=True,
                                                                sampler=torch.utils.data.WeightedRandomSampler(
                                                                    selected_examples, len(selected_examples)))
            train_sel(args, scheduler, model, model_ema, uns_contrast, queue, device, train_loader,
                      train_selected_loader, optimizer, epoch, selected_pairs)

        if (epoch >= args.warmup_epoch):
            print('######## Pair-wise selection ########')
            selected_examples, selected_pairs = pair_selection(args, model, device, train_loader, test_loader,
                                                               epoch)

        print('Epoch time: {:.2f} seconds\n'.format(time.time() - st))

        test_eval(args, model, device, test_loader)

        acc, acc5 = kNN(args, epoch, model, None, train_loader, test_loader, args.k_val if args.k_val<200 else 200, 0.1, True)
        print('\nEpoch {}, acc {} \n'.format(epoch, acc))

        print('KNN top-1 precion: {:.4f}'.format(acc * 100.))

        test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
                                      'Test', True, plt_cm=plt_cm, plt_lables=False)

        accs.append(test_results['acc'])
        f1s.append(test_results['f1_weighted'])

        # if (epoch % 10 == 0):
        #     save_model(model, optimizer, args, epoch, exp_path + "/Sel-CL_model.pth")
        #     np.save(res_path + '/' + 'selected_examples_train.npy', selected_examples.data.cpu().numpy())

        # log_file.flush()
    ######################################################################################################
    # Fine-tune model
    # #
    if args.fine_tune:
        # clean_idx = np.load(res_path + "/selected_examples_train.npy")
        clean_idx = selected_examples.data.cpu().numpy()
        exp_path = exp_path + "/plus/"
        res_path = res_path + "/plus/"
        if not os.path.isdir(res_path):
            os.makedirs(res_path)

        if not os.path.isdir(exp_path):
            os.makedirs(exp_path)

        # __console__ = sys.stdout
        # log_file = open(res_path + "results.log", 'a')
        # sys.stdout = log_file
        print(args)

        if args.ReInitializeClassif == 1:
            model.linear2 = nn.Linear(512, args.num_classes).to(device)

        milestones = args.M

        optimizer = optim.SGD(model.parameters(), lr=args.ft_lr, momentum=args.momentum, weight_decay=args.wd)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


        new_train_dataset = TensorDataset(torch.from_numpy(x_train[clean_idx==1]).float(), torch.from_numpy(Y_train[clean_idx==1]).long(),
                                  torch.from_numpy(np.arange(len(Y_train[clean_idx==1]))),torch.from_numpy(Y_train_clean[clean_idx==1]))


        train_loader = DataLoader(new_train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                       num_workers=args.num_workers)

        for epoch in range(args.ft_initial_epoch, args.ft_epoch + 1):
            st = time.time()
            print("=================>    ", args.experiment_name, args.ni[0])
            scheduler.step()
            train_acc,train_loss,model=train_model(train_loader,  model, optimizer, epoch, args)
            print('Epoch time: {:.2f} seconds\n'.format(time.time() - st))
            print('\nEpoch:{}, train_acc:{}, train_loss:{}'.format(epoch,train_acc,train_loss))

            test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
                                          'Test', True, plt_cm=plt_cm, plt_lables=False)
            accs.append(test_results['acc'])
            f1s.append(test_results['f1_weighted'])

    ######################################################################################################

    valid_results = evaluate_class(model, x_valid, Y_valid, Y_valid_clean, valid_loader, ni, saver,
                                   'CNN', 'Valid', True, plt_cm=plt_cm, plt_lables=False)
    test_results = evaluate_class(model, x_test, Y_test, None, test_loader, ni, saver, 'CNN',
                                  'Test', True, plt_cm=plt_cm, plt_lables=False)

    test_results['max_valid_acc'] = np.max(accs)
    test_results['max_valid_acc_epoch'] = np.argmax(accs)
    test_results['avg_last_ten_valid_acc'] = np.mean(accs[-10:])
    test_results['max_valid_f1'] = np.max(f1s)
    test_results['max_valid_f1_epoch'] = np.argmax(f1s)
    test_results['avg_last_ten_valid_f1'] = np.mean(f1s[-10:])

    torch.cuda.empty_cache()
    return valid_results, test_results

def train_model(data_loader, model, optimizer,epoch=None,
                args=None):
    criterion = nn.CrossEntropyLoss(reduce=False)
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

        _, out = model(x)
        model_loss = criterion(out, y_hat).sum()

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

    return avg_accuracy / global_step, avg_loss / global_step, model
