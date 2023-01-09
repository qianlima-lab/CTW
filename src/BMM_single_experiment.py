import argparse
import logging
import os
import shutil
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from pyts import datasets
import time

sys.path.append(os.path.dirname(sys.path[0]))
from src.utils.utils import create_synthetic_dataset
from src.utils.global_var import OUTPATH
from src.utils.plotting_utils import plot_label_insight
from src.utils.saver import Saver
from src.utils.training_helper_BMM import main_wrapper
from src.ucr_data.load_ucr_pre import load_ucr
from src.uea_data.load_uea_pre import load_uea
from src.utils.log_utils import StreamToLogger,get_logger,create_logfile

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns


######################################################################################################

def parse_args():
    # TODO: make external configuration file -json or similar.
    """
    Parse arguments
    """
    # Add global parameters
    parser = argparse.ArgumentParser(description='BMM single experiment on UCR datasets.')

    # Synth Data
    parser.add_argument('--dataset', type=str, default='Plane', help='UCR datasets')

    parser.add_argument('--ni', type=float, default=0.5, help='label noise ratio')
    parser.add_argument('--label_noise', type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                                   'number as str for time-dependent noise')

    parser.add_argument('--M', type=int, nargs='+', default=[60, 120, 180, 240])
    parser.add_argument('--reg_term', type=float, default=1,
                        help="Parameter of the regularization term, default: 0.")
    parser.add_argument('--alpha', type=float, default=32,
                        help='alpha parameter for the mixup distribution, default: 32')

    parser.add_argument('--correct', type=str, default='MixUp-BMM', help='Correct labels',choices=['None', 'Mixup', 'MixUp-BMM'])  ##

    parser.add_argument('--Mixup', type=str, default='Dynamic', choices=['None', 'Static', 'Dynamic'],
                        help="Type of bootstrapping. Available: 'None' (deactivated)(default), \
                                'Static' (as in the paper), 'Dynamic' (BMM to mix the smaples, will use decreasing softmax), default: None")
    parser.add_argument('--BootBeta', type=str, default='Hard', choices=['None', 'Hard', 'Soft'],
                        help="Type of Bootstrapping guided with the BMM. Available: \
                        'None' (deactivated)(default), 'Hard' (Hard bootstrapping), 'Soft' (Soft bootstrapping), default: Hard")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status, default: 10')

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed - only affects Network init')

    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=32)

    # CNN
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--plt_embedding', action='store_true', default=False)
    parser.add_argument('--plt_loss_hist', action='store_true', default=False)
    parser.add_argument('--plt_cm', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=False, help='Matplotlib backend')
    parser.add_argument('--manual_seeds', type=int, nargs='+', default=[37, 118, 337, 815, 19],
                        help='manual_seeds for five folds')
    parser.add_argument('--from_ucr', type=int, default=0, help='begin from which dataset')
    parser.add_argument('--end_ucr', type=int, default=128, help='end at which dataset')
    parser.add_argument('--ucr', type=int, default=0, help='if 128, run all ucr datasets')
    parser.add_argument('--model', choices=['BMM','sigua'], default='BMM')
    parser.add_argument('--aug', choices=['NoAug'], default='NoAug')
    parser.add_argument('--cuda_device', type=int, default=0, help='choose the cuda devcie')
    parser.add_argument('--basicpath', type=str, default='', help='basic path')
    parser.add_argument('--outfile', type=str, default='Mixup_BMM.csv', help='filename')
    parser.add_argument('--debug', action='store_true', default=False,help='')

    args = parser.parse_args()
    torch.cuda.set_device(args.cuda_device)
    return args


######################################################################################################
def main(args, dataset_name=None):

    # Declare saver object
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  hierarchy=os.path.join(args.dataset),args=args)

    ######################################################################################################
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
    args.dataset = dataset_name
    if args.dataset=='synthesis':
        X, Y=create_synthetic_dataset(ts_n=800)
    elif args.dataset in datasets.uea_dataset_list():
        X, Y = load_uea(args.dataset)
    else:
        X, Y = load_ucr(args.dataset)

    skf = StratifiedKFold(n_splits=5)
    id_acc = 0
    seeds_i = -1
    seeds = args.manual_seeds
    starttime = time.time()
    for trn_index, test_index in skf.split(X, Y):
        id_acc = id_acc + 1
        seeds_i = seeds_i + 1
        print("id_acc = ", id_acc, trn_index.shape, test_index.shape)
        x_train = X[trn_index]
        x_test = X[test_index]
        Y_train_clean = Y[trn_index]
        Y_test_clean = Y[test_index]

        Y_valid_clean = Y_test_clean.copy()
        x_valid = x_test.copy()

        batch_size = min(x_train.shape[0] // 10, args.batch_size)
        if x_train.shape[0] % batch_size == 1:
            batch_size += -1
        print('Batch size: ', batch_size)
        args.batch_size = batch_size
        args.test_batch_size = batch_size

        ################################
        len_x_train = len(x_train)
        batches_per_epoch = len_x_train / batch_size
        if batches_per_epoch<=10:
            args.warmup=min(int(150/batches_per_epoch),25)
        elif batches_per_epoch<=20:
            args.warmup=int(300/batches_per_epoch)
        elif batches_per_epoch<=30:
            args.warmup=int(500/batches_per_epoch)
        else:
            args.warmup=min(int(700/batches_per_epoch),25)
        ###########################
        saver.make_log(**vars(args))
        # plot_label_insight(x_train, Y_train_clean, saver=saver)

        ######################################################################################################
        df_results = main_wrapper(args, x_train, x_valid, x_test, Y_train_clean, Y_valid_clean, Y_test_clean,
                                  saver,seed=seeds[seeds_i])

        five_test_acc.append(df_results["acc"])
        five_test_f1.append(df_results["f1_weighted"])
        five_avg_last_ten_test_acc.append(df_results["avg_last_ten_test_acc"])
        five_avg_last_ten_test_f1.append(df_results["avg_last_ten_test_f1"])

    # print('Save results')
    # df_results.to_csv(os.path.join(saver.path, 'results.csv'), sep=',', index=False)
    endtime = time.time()
    seconds = endtime - starttime
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    deltatime = "%d:%d:%d" % (h, m, s)
    result_evalution["deltatime"] = deltatime
    result_evalution["dataset_name"] = args.dataset
    result_evalution["avg_five_test_acc"] = round(np.mean(five_test_acc), 4)
    result_evalution["std_five_test_acc"] = round(np.std(five_test_acc), 4)
    result_evalution["avg_five_test_f1"] = round(np.mean(five_test_f1), 4)
    result_evalution["std_five_test_f1"] = round(np.std(five_test_f1), 4)
    result_evalution["avg_five_avg_last_ten_test_acc"] = round(np.mean(five_avg_last_ten_test_acc), 4)
    result_evalution["avg_five_avg_last_ten_test_f1"] = round(np.mean(five_avg_last_ten_test_f1), 4)

    return result_evalution


######################################################################################################
if __name__ == '__main__':
    args = parse_args()

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
        ucr = ['ArrowHead', 'CBF', 'FaceFour', 'MelbournePedestrian', 'OSULeaf', 'Plane', 'Symbols', 'Trace',
               'Epilepsy', 'NATOPS', 'EthanolConcentration', 'FaceDetection', 'FingerMovements']
        # ucr = ['FaceDetection', 'FingerMovements']

    for dataset_name in ucr:
        args = parse_args()
        args.dataset = dataset_name
        args.basicpath = basicpath

        df_results = main(args, dataset_name=dataset_name)
        result_value.append(df_results)

        print("result_value = ", result_value)

        datestr = time.strftime(('%Y%m%d'))

        if args.label_noise == -1:
            label_noise = 'inst'
        elif args.label_noise == 0:
            label_noise = 'sym'
        else:
            label_noise = "asym"

        path = os.path.abspath(os.path.join(basicpath, 'statistic_results', args.outfile))
        pd.DataFrame(result_value).to_csv(path)
