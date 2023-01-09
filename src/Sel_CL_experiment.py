import torch.backends.cudnn as cudnn
from torchvision import datasets
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append('../Sel-CL_utils')
from src.utils.Sel_CL_utils.utils_noise import *
# from src.models.preact_resnet import *


import argparse
import logging

import time

sys.path.append(os.path.dirname(sys.path[0]))
import shutil

import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyts import datasets

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
from sklearn.model_selection import StratifiedKFold

sys.path.append("..")
print(os.getcwd())

from src.utils.utils import create_synthetic_dataset
from src.utils.global_var import OUTPATH
from src.utils.saver import Saver
from src.utils.training_helper_Sel_CL import main_wrapper_Sel_CL
from src.ucr_data.load_ucr_pre import load_ucr
from src.uea_data.load_uea_pre import load_uea
from src.utils.log_utils import StreamToLogger,get_logger,create_logfile

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns


def parse_args(command=None):
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('--warmup_way', type=str, default="uns", help='uns, sup')
    parser.add_argument('--warmup-epoch', type=int, default=1, help='warmup epoch')
    parser.add_argument('--lr', '--base-learning-rate', '--base-lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--lr-warmup-epoch', type=int, default=1, help='warmup epoch')
    parser.add_argument('--lr_warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr_decay_epochs', type=int, default=[125, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--initial_epoch', type=int, default=1, help="Star training at initial_epoch")

    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--cuda_device', type=int, default=0, help='GPU to select')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of in-distribution classes')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset', type=str, default='CBF', help='')
    parser.add_argument('--noise_type', default='asymmetric', help='symmetric or asymmetric')
    parser.add_argument('--train_root', default='./dataset', help='root for train data')
    parser.add_argument('--out', type=str, default='./results/Sel_CL_out', help='Directory of the output')
    parser.add_argument('--experiment_name', type=str, default='Proof',
                        help='name of the experiment (for the output files)')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')

    parser.add_argument('--network', type=str, default='FCN', help='Network architecture')
    parser.add_argument('--headType', type=str, default="Linear", help='Linear, NonLinear')
    parser.add_argument('--low_dim', type=int, default=128, help='Size of contrastive learning embedding')
    parser.add_argument('--seed_initialization', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--DA', type=str, default="complex", help='Choose simple or complex data augmentation')

    parser.add_argument('--alpha_m', type=float, default=1.0, help='Beta distribution parameter for mixup')
    parser.add_argument('--alpha_moving', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--alpha', type=float, default=0.5, help='example selection th')
    parser.add_argument('--beta', type=float, default=0.5, help='pair selection th')
    parser.add_argument('--uns_queue_k', type=int, default=10000, help='uns-cl num negative sampler')
    parser.add_argument('--uns_t', type=float, default=0.1, help='uns-cl temperature')
    parser.add_argument('--sup_t', default=0.1, type=float, help='sup-cl temperature')
    parser.add_argument('--sup_queue_use', type=int, default=1, help='1: Use queue for sup-cl')
    parser.add_argument('--sup_queue_begin', type=int, default=3, help='Epoch to begin using queue for sup-cl')
    parser.add_argument('--queue_per_class', type=int, default=100,
                        help='Num of samples per class to store in the queue. queue size = queue_per_class*num_classes*2')
    parser.add_argument('--aprox', type=int, default=1,
                        help='Approximation for numerical stability taken from supervised contrastive learning')
    parser.add_argument('--lambda_s', type=float, default=0.01, help='weight for similarity loss')
    parser.add_argument('--lambda_c', type=float, default=1, help='weight for classification loss')
    parser.add_argument('--k_val', type=int, default=250, help='k for k-nn correction')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")

    # new
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=False, help='Matplotlib backend')
    parser.add_argument('--manual_seeds', type=int, nargs='+', default=[37, 118, 337, 815, 19],
                        help='manual_seeds for five folds cross varidation')
    parser.add_argument('--num_training_samples',type=int,default=0,help='num of trainging samples')
    parser.add_argument('--sample_len', type=int,default=0)
    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--aug', choices=['GNoise','NoAug','Oversample','Convolve','Crop','Drift','TimeWarp','Mixup'],default='NoAug')
    parser.add_argument('--ucr', type=int, default=0, help='if 128, run all ucr datasets')
    parser.add_argument('--basicpath', type=str, default='', help='basic path')
    parser.add_argument('--model', default='Sel_CL', choices=['Sel_CL'])
    parser.add_argument('--label_noise', type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                                   'number as str for time-dependent noise')
    parser.add_argument('--warmup',type=int,default=10,help='warmup epochs' )
    parser.add_argument('--from_ucr', type=int, default=0, help='begin from which dataset')
    parser.add_argument('--end_ucr', type=int, default=128, help='end at which dataset')

    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--num_gradual', type=int, default=100)

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--seed', type=int, default=0, help='RNG seed - only affects Network init')

    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=512)

    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)
    parser.add_argument('--ni', type=float, default=0.5, help='label noise ratio')
    parser.add_argument('--startLabelCorrection', type=int, default=30, help='Epoch to start label correction')
    parser.add_argument('--ReInitializeClassif', type=int, default=0, help='Enable predictive label correction')
    parser.add_argument('--ft_initial_epoch', type=int, default=1, help='fine tune args')
    parser.add_argument('--ft_epoch', type=int, default=1, help='fine tune args')
    parser.add_argument('--fine_tune', action='store_true', default=False, help='if fine tune')
    parser.add_argument('--ft_lr', type=float, default=0.0001, help='fine tune args')
    parser.add_argument('--outfile', type=str, default='Sel_CL.csv', help='filename')
    parser.add_argument('--debug', action='store_true', default=False,help='')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        torch.cuda.set_device(args.cuda_device)
    return args


def main(args, dataset_name=None):

    # Declare saver object
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  hierarchy=os.path.join(args.dataset),args=args)

    ######################################################################################################
    print(args)
    print()

    ######################################################################################################
    SEED = args.seed
    # TODO: implement multi device and different GPU selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    if args.headless:
        print('Setting Headless support')
        plt.switch_backend('Agg')
    else:
        backend = 'Qt5Agg'
        print('Swtiching matplotlib backend to', backend)
        # plt.switch_backend(backend)
    print()

    ######################################################################################################
    # Data
    print('*' * shutil.get_terminal_size().columns)
    print('UCR Dataset: {}'.format(args.dataset).center(columns))
    print('*' * shutil.get_terminal_size().columns)
    print()

    five_test_acc = []
    five_test_f1 = []
    five_avg_last_ten_test_acc = []
    five_avg_last_ten_test_f1 = []

    result_evalution = dict()

    # X, Y = load_data(args.dataset)
    if args.dataset=='synthesis':
        X, Y=create_synthetic_dataset(ts_n=800)
    elif args.dataset in datasets.uea_dataset_list():
        X, Y = load_uea(args.dataset)
    else:
        X, Y = load_ucr(args.dataset)
    classes = len(np.unique(Y))
    args.num_classes = classes

    skf = StratifiedKFold(n_splits=5)
    id_acc = 0
    seeds_i = -1
    seeds = args.manual_seeds
    starttime = time.time()
    for trn_index, test_index in skf.split(X, Y):
        args.num_training_samples = len(trn_index)
        args.sample_len = X.shape[1]
        seeds_i = seeds_i + 1
        id_acc = id_acc + 1
        print("id_acc = ", id_acc, trn_index.shape, test_index.shape)
        x_train = X[trn_index]
        x_test = X[test_index]
        Y_train_clean = Y[trn_index]
        Y_test_clean = Y[test_index]
        args.k_val = min(np.median(np.bincount(Y_train_clean)).astype(int),args.k_val)

        args.uns_queue_k = int(X.shape[0]/5)
        args.queue_per_class = int(args.uns_queue_k/args.num_classes)

        batch_size = min(x_train.shape[0] // 10, args.batch_size)
        if x_train.shape[0] % batch_size == 1:
            batch_size += -1
        print('Batch size: ', batch_size)
        args.batch_size = batch_size
        args.test_batch_size = batch_size
        if args.batch_size<=20:
            args.num_workers=0
        # args.batch_size = 2906
        # ##########################
        len_x_train=len(x_train)

        if len_x_train<=1000:
            args.warmup=30
        elif len_x_train<=3000:
            args.warmup=15
        else:
            args.warmup=10
        # args.warmup=1
        # ##########################
        saver.make_log(**vars(args))
        ######################################################################################################

        df_results = main_wrapper_Sel_CL(args, x_train, x_test, Y_train_clean, Y_test_clean,
                                         saver,seeds=[seeds[seeds_i]])

        five_test_acc.append(df_results["acc"])
        five_test_f1.append(df_results["f1_weighted"])
        five_avg_last_ten_test_acc.append(df_results["avg_last_ten_test_acc"])
        five_avg_last_ten_test_f1.append(df_results["avg_last_ten_test_f1"])

    # print('Save results')
    # df_results.to_csv(os.path.join(saver.path, 'results.csv'), sep=',', index=False)
    endtime = time.time()
    result_evalution["dataset_name"] = args.dataset
    result_evalution["avg_five_test_acc"] = round(np.mean(five_test_acc), 4)
    result_evalution["std_five_test_acc"] = round(np.std(five_test_acc), 4)
    result_evalution["avg_five_test_f1"] = round(np.mean(five_test_f1), 4)
    result_evalution["std_five_test_f1"] = round(np.std(five_test_f1), 4)
    result_evalution["avg_five_avg_last_ten_test_acc"] = round(np.mean(five_avg_last_ten_test_acc), 4)
    result_evalution["avg_five_avg_last_ten_test_f1"] = round(np.mean(five_avg_last_ten_test_f1), 4)

    seconds = endtime - starttime
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    deltatime = "%d:%d:%d"%(h,m,s)
    result_evalution["deltatime"]=deltatime
    return result_evalution


if __name__ == '__main__':
    args = parse_args()
    print('\nargs\n:', args)
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    basicpath = os.path.dirname(father_path)

    # Logging setting
    if not args.debug:  # if not debug, no log.
        logger = get_logger(logging.INFO, args.debug, args=args, filename='logfile.log')
        __stderr__ = sys.stderr  #
        sys.stderr = open(create_logfile(args, 'error.log'), 'a')
        __stdout__ = sys.stdout
        sys.stdout = StreamToLogger(logger, logging.INFO)

    print("father_path = ", father_path)
    result_value = []

    if args.ucr == 128:
        ucr = datasets.ucr_dataset_list()[args.from_ucr:args.end_ucr]
    else:
        ucr=['ArrowHead','CBF','FaceFour','MelbournePedestrian','OSULeaf','Plane','Symbols','Trace',
             'Epilepsy','NATOPS','EthanolConcentration', 'FaceDetection', 'FingerMovements']
        # ucr=['EthanolConcentration', 'FaceDetection', 'FingerMovements']

    for dataset_name in ucr:
        args = parse_args()  # restart
        args.basicpath = basicpath
        args.dataset = dataset_name
        df_results = main(args, dataset_name)
        result_value.append(df_results)

        print("result_value = ", result_value)

        if args.label_noise == -1:
            label_noise = 'inst'
        elif args.label_noise == 0:
            args.noise_type = 'symmetric'
            label_noise = 'sym'
        else:
            args.noise_type = 'asymmetric'
            label_noise = "asym"

        path = os.path.abspath(os.path.join(basicpath, 'statistic_results', args.outfile))
        pd.DataFrame(result_value).to_csv(path)
