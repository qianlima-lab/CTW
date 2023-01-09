import argparse
import logging
import os
import shutil
import sys
sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append("..")
print(os.getcwd())

import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from pyts import datasets

from src.utils.training_helper_SREA import main_wrapper
from src.utils.global_var import OUTPATH
from src.utils.plotting_utils import plot_label_insight
from src.utils.saver import Saver
from src.ucr_data.load_ucr_pre import load_ucr
from src.uea_data.load_uea_pre import load_uea
from src.utils.log_utils import StreamToLogger,get_logger,create_logfile

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns


######################################################################################################

def parse_args():
    # Add global parameters
    parser = argparse.ArgumentParser(
        description='SREA')

    parser.add_argument('--dataset', type=str, default='CBF', help='UCR datasets')

    parser.add_argument('--ni', type=float, default=0.30, help='label noise ratio')
    parser.add_argument('--label_noise',type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                         'number as str for time-dependent noise')

    parser.add_argument('--M', type=int, nargs='+', default=[20, 40, 60, 80], help='Scheduler milestones')
    parser.add_argument('--abg', type=float, nargs='+',
                        help='Loss function coefficients. a (alpha) = AE, b (beta) = classifier, g (gamma) = clusterer',
                        default=[1, 1, 1])
    parser.add_argument('--class_reg', type=int, default=1, help='Distribution regularization coeff')
    parser.add_argument('--entropy_reg', type=int, default=0., help='Entropy regularization coeff')

    parser.add_argument('--correct', action='store_true', default=True,
                        help='Correct labels. Set to false to not correct labels.')
    parser.add_argument('--track', type=int, default=5, help='Number or past predictions snapshots')
    parser.add_argument('--init_centers', type=int, default=1, help='Initialize cluster centers. Warm up phase.')
    parser.add_argument('--delta_start', type=int, default=10, help='Start re-labeling phase')
    parser.add_argument('--delta_end', type=int, default=30,
                        help='Begin fine-tuning phase')

    parser.add_argument('--preprocessing', type=str, default='StandardScaler',
                        help='Any available preprocessing method from sklearn.preprocessing')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--seed', type=int, default=0, help='Initial RNG seed. Only for reproducibility')

    parser.add_argument('--classifier_dim', type=int, default=128, help='Dimension of final classifier')
    parser.add_argument('--embedding_size', type=int, default=32, help='Dimension of embedding')

    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False,
                        help='Suppress screen print, keep log.txt')
    parser.add_argument('--plt_loss', action='store_true', default=False, help='plot loss function each epoch')
    parser.add_argument('--plt_embedding', action='store_true', default=False, help='plot embedding representation')
    parser.add_argument('--plt_loss_hist', action='store_true', default=False,
                        help='plot loss history for clean and mislabled samples')
    parser.add_argument('--plt_cm', action='store_true', default=False, help='plot confusion matrix')
    parser.add_argument('--plt_recons', action='store_true', default=False, help='plot AE reconstructions')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Matplotlib backend. Set true if no display connected.')
    parser.add_argument('--num_training_samples',type=int,default=0,help='num of trainging samples')
    parser.add_argument('--cuda_device', type=int, default=0, help='choose the cuda devcie')
    parser.add_argument('--manual_seeds', type=int, nargs='+', default=[37, 118, 337, 815, 19],
                        help='manual_seeds for five folds cross varidation')

    parser.add_argument('--warmup', type=int, default=10, help='warmup epochs')
    parser.add_argument('--ucr', type=int, default=0, help='if 128, run all ucr datasets')
    parser.add_argument('--model', choices=['SREA'],default='SREA')
    parser.add_argument('--aug', choices=['NoAug'],default='NoAug')
    parser.add_argument('--from_ucr', type=int, default=0, help='begin from which dataset')
    parser.add_argument('--end_ucr', type=int, default=128, help='end at which dataset')
    parser.add_argument('--basicpath', type=str, default='', help='basic path')
    parser.add_argument('--outfile', type=str, default='SREA.csv', help='filename')
    parser.add_argument('--debug', action='store_true', default=False,help='')

    args = parser.parse_args()
    torch.cuda.set_device(args.cuda_device)
    return args


######################################################################################################
def main(args,dataset_name=None):

    print(args)
    print()

    ######################################################################################################

    if args.headless:
        print('Setting Headless support')
        plt.switch_backend('Agg')
    else:
        backend = 'Qt5Agg'
        print('Swtiching matplotlib backend to', backend)
        # plt.switch_backend(backend)
    print()

    ######################################################################################################
    # LOG STUFF
    # Declare saver object
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  hierarchy=os.path.join(args.dataset),args=args)

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

    if args.dataset in datasets.ucr_dataset_list():
        X, Y = load_ucr(args.dataset)
    else:
        X, Y = load_uea(args.dataset)

    skf = StratifiedKFold(n_splits=5)
    id_acc = 0
    seeds_i = -1
    seeds = args.manual_seeds
    starttime = time.time()
    for trn_index, test_index in skf.split(X, Y):
        args.num_training_samples = len(trn_index)
        seeds_i = seeds_i + 1
        id_acc = id_acc + 1
        print("id_acc = ", id_acc, trn_index.shape, test_index.shape)
        x_train = X[trn_index]
        x_test = X[test_index]
        Y_train_clean = Y[trn_index]
        Y_test_clean = Y[test_index]

        batch_size = min(x_train.shape[0] // 10, args.batch_size)
        if x_train.shape[0] % batch_size == 1:
            batch_size += -1
        print('Batch size: ', batch_size)
        args.batch_size = batch_size
        args.test_batch_size = batch_size

        # ##########################
        len_x_train = len(x_train)

        if len_x_train <= 1000:
            args.warmup = 30
        elif len_x_train <= 3000:
            args.warmup = 15
        else:
            args.warmup = 10
        ###########################
        saver.make_log(**vars(args))
        plot_label_insight(x_train, Y_train_clean, saver=saver)

        ######################################################################################################
        df_results = main_wrapper(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=seeds[seeds_i],dataset_name=dataset_name)

        print("df_results = ", df_results)
        print("df_results = ", df_results["acc"])
        print('Save results')

        five_test_acc.append(df_results["acc"])
        five_test_f1.append(df_results["f1_weighted"])
        five_avg_last_ten_test_acc.append(df_results["avg_last_ten_test_acc"] / 100)
        five_avg_last_ten_test_f1.append(df_results["avg_last_ten_test_f1"])

    endtime = time.time()
    result_evalution["dataset_name = "] = args.dataset
    result_evalution["avg_five_test_acc"] = round(np.mean(five_test_acc), 4)
    result_evalution["std_five_test_acc"] = round(np.std(five_test_acc), 4)
    result_evalution["avg_five_test_f1"] = round(np.mean(five_test_f1), 4)
    result_evalution["std_five_test_f1"] = round(np.std(five_test_f1), 4)
    result_evalution["avg_five_avg_last_ten_test_acc"] = round(np.mean(five_avg_last_ten_test_acc), 4)
    result_evalution["avg_five_avg_last_ten_test_f1"] = round(np.mean(five_avg_last_ten_test_f1), 4)

    seconds = endtime - starttime
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    deltatime = "%d:%d:%d" % (h, m, s)
    result_evalution["deltatime"] = deltatime
    return result_evalution


######################################################################################################
if __name__ == '__main__':
    args = parse_args()

    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

    # Logging setting
    if not args.debug:  # if not debug, no log.
        logger = get_logger(logging.INFO, args.debug, args=args, filename='logfile.log')
        __stderr__ = sys.stderr  #
        sys.stderr = open(create_logfile(args, 'error.log'), 'a')
        __stdout__ = sys.stdout
        sys.stdout = StreamToLogger(logger, logging.INFO)

    print("father_path = ", father_path)
    basicpath = os.path.dirname(father_path)
    result_value = []
    if args.ucr == 128:
        ucr = datasets.ucr_dataset_list()[args.from_ucr:args.end_ucr]
    else:
        ucr = ['ArrowHead', 'CBF', 'FaceFour', 'MelbournePedestrian', 'OSULeaf', 'Plane', 'Symbols', 'Trace',
               'Epilepsy', 'NATOPS', 'EthanolConcentration', 'FaceDetection', 'FingerMovements']

    for dataset_name in ucr:
        args = parse_args()
        args.dataset = dataset_name
        args.basicpath = basicpath
        df_results = main(args, dataset_name=dataset_name)
        result_value.append(df_results)

        print("result_value = ", result_value)

        if args.label_noise == -1:
            label_noise = 'inst'
        elif args.label_noise == 0:
            label_noise = 'sym'
        else:
            label_noise = "asym"

        path = os.path.abspath(os.path.join(basicpath, 'statistic_results', args.outfile))
        pd.DataFrame(result_value).to_csv(path)
