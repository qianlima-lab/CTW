import argparse
import logging
import os
import sys
import time

sys.path.append(os.path.dirname(sys.path[0]))
import shutil

import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyts import datasets
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True
from sklearn.model_selection import StratifiedKFold


# sys.path.append("..")

from src.utils.log_utils import StreamToLogger,get_logger,create_logfile
from src.utils.utils import create_synthetic_dataset
from src.utils.global_var import BASE_PATH,OUTPATH
from src.utils.saver import Saver
from src.utils.training_helper_coteaching import main_wrapper
from src.utils.training_helper_single_model import main_wrapper_single_model
from src.utils.training_helper_CTW import main_wrapper_CTW
from src.utils.training_helper_dividemix import main_wrapper_dividemix
from src.ucr_data.load_ucr_pre import load_ucr
from src.uea_data.load_uea_pre import load_uea

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
    parser = argparse.ArgumentParser(description='coteaching single experiment')

    # Synth Data
    parser.add_argument('--dataset', type=str, default='CBF', help='UCR datasets')
    parser.add_argument('--outfile', type=str, default='CTW.csv', help='name of output file')
    parser.add_argument('--ni', type=float, default=0.5, help='label noise ratio')
    parser.add_argument('--label_noise', type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                                   'number as str for time-dependent noise')

    parser.add_argument('--M', type=int, nargs='+', default=[20, 40, 60, 80])
    parser.add_argument('--reg_term', type=float, default=1,
                        help="Parameter of the regularization term, default: 0.")
    parser.add_argument('--alpha', type=float, default=32,
                        help='alpha parameter for the mixup distribution, default: 32')

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--hidden_activation', type=str, default='nn.ReLU()')
    parser.add_argument('--num_gradual', type=int, default=100)

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed - only affects Network init')

    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=32)

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
    parser.add_argument('--beta',type=float,nargs='+',default=[0.,3.],help='the coefficient of model_loss2')
    parser.add_argument('--warmup',type=int,default=10,help='warmup epochs' )

    parser.add_argument('--model',choices=['co_teaching','co_teaching_mloss',
                                           'sigua', 'single_ae_aug_after_sel','single_aug','single_sel','vanilla',
                                           'single_aug_after_sel','single_ae_sel','single_ae','single_ae_aug',
                                           'single_ae_aug_sel_allaug','single_ae_aug_before_sel','dividemix', 'CTW'],
                        default='CTW')
    parser.add_argument('--manual_seeds', type=int, nargs='+', default=[37, 118, 337, 815, 19],
                        help='manual_seeds for five folds cross varidation')
    parser.add_argument('--label_correct', action='store_true', default=False, help='if correct label')
    parser.add_argument('--num_training_samples',type=int,default=0,help='num of trainging samples')
    parser.add_argument('--loss', type=str, default='cores', help='type of loss function')
    parser.add_argument('--mixup', action='store_true', default=False, help='manifold mixup if or not')
    parser.add_argument('--mean_loss_len', type=int,default=1,help='the length of mean loss')
    parser.add_argument('--gamma', type=float, default=0.5, help='the weight of current sample loss in mean_loss_sel method')
    parser.add_argument('--arg_interval', type=int, default=1,
                        help='the batch-interval for augmentation in batch')
    parser.add_argument('--cuda_device', type=int, default=0, help='choose the cuda devcie')
    parser.add_argument('--aug', choices=['GNoise','NoAug','Oversample','Convolve','Crop','Drift','TimeWarp','Mixup'], default='NoAug')
    parser.add_argument('--sample_len', type=int,default=0)
    parser.add_argument('--ucr', type=int, default=0,help='if 128, run all ucr datasets')
    parser.add_argument('--basicpath', type=str, default='', help='basic path')
    parser.add_argument('--plot_tsne', action='store_true', default=False, help='if plot t-sne or not')
    parser.add_argument('--nbins', type=int, default=0, help='number of class')
    parser.add_argument('--save_model', action='store_true', default=False, help='if save model or not')
    parser.add_argument('--from_ucr', type=int, default=0, help='begin from which dataset')
    parser.add_argument('--end_ucr', type=int, default=128, help='end at which dataset')
    parser.add_argument('--sel_method', type=int, default=3,choices=[0,1,2,3,4],
                        help='''0: select ratio is known (co-teaching, sigua);
                                1,2: select confident samples class by class;
                                3: select w/ EPS
                                4: select w/o EPS''')
    parser.add_argument('--tsne_during_train', action='store_true', default=False, help='if plot tsne during training or not')
    parser.add_argument('--tsne_epochs', type=int, nargs='+', default=[49, 99, 149, 199, 249,299],
                        help='manual_seeds for five folds cross varidation')

    parser.add_argument('--augMSE', action='store_true', default=False, help='if use MSE on aug or not')
    parser.add_argument('--bad_weight', type=float, default=1e-3,help='for sigua')
    parser.add_argument('--aug_ae', action='store_true', default=False, help='if reconstruct augmented samples or not')
    parser.add_argument('--window', type=str, choices=['single', 'all'], default='all',
                        help='single_train/single_test: only plot training/test data; all: plot all data ')
    parser.add_argument('--L_aug_coef', type=float, default=1.,
                        help='the coefficient of L_aug')
    parser.add_argument('--L_rec_coef', type=float, default=1.,
                        help='the coefficient of L_rec')
    parser.add_argument('--confcsv', type=str, default=None,
                        help='the file of saving conf_num')
    parser.add_argument('--whole_data_select', action='store_true', default=False,
                        help='if select from whole data')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
    parser.add_argument('--plt_loss_density', action='store_true', default=False,
                        help='if plot loss density')
    parser.add_argument('--standardization_choice', type=str, choices=['z-score', 'min-max'], default='z-score',
                        help='choose the method of standardization')
    parser.add_argument('--debug', action='store_true', default=False,help='')

    # Add parameters for each particular network

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device==torch.device('cuda'):
        torch.cuda.set_device(args.cuda_device)

    return args

######################################################################################################
def main(args, dataset_name=None):

    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  hierarchy=os.path.join(args.dataset),args=args)

    if args.plot_tsne:
        args.save_model=True

    ######################################################################################################
    print(f'{args}')

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
        print(f'Swtiching matplotlib backend to {backend}')
        # plt.switch_backend(backend)

    ######################################################################################################
    # Data
    print('*' * shutil.get_terminal_size().columns)
    print('UCR Dataset: {}'.format(args.dataset).center(columns))
    print('*' * shutil.get_terminal_size().columns)

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
    args.nbins = classes

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
        print(f"id_acc = {id_acc}, {trn_index.shape}, {test_index.shape}", )
        x_train = X[trn_index]
        x_test = X[test_index]
        Y_train_clean = Y[trn_index]
        Y_test_clean = Y[test_index]

        batch_size = min(x_train.shape[0] // 10, args.batch_size)
        if x_train.shape[0] % batch_size == 1:
            batch_size += -1
        print(f'Batch size: {batch_size}')
        args.batch_size = batch_size
        args.test_batch_size = batch_size

        # ##########################
        len_x_train=len(x_train)

        if len_x_train <= 1000:
            args.warmup = 30
        elif len_x_train <= 3000:
            args.warmup = 15
        else:
            args.warmup = 10
        # ##########################
        saver.make_log(**vars(args))
        ######################################################################################################

        if args.model in ['CTW']:
            df_results = main_wrapper_CTW(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=seeds[seeds_i])
        elif args.model in ['single_ae_aug_after_sel', 'single_aug', 'single_sel', 'vanilla', 'single_ae_aug_sel_allaug',
                          'single_aug_after_sel', 'single_ae_sel', 'single_ae', 'single_ae_aug','single_ae_aug_before_sel']:
            df_results = main_wrapper_single_model(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=seeds[seeds_i])
        elif args.model in ['dividemix']:
            df_results = main_wrapper_dividemix(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=seeds[seeds_i])
        else: # co-teaching, sigua
            df_results = main_wrapper(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,seed=seeds[seeds_i])

        five_test_acc.append(df_results["acc"])
        five_test_f1.append(df_results["f1_weighted"])
        five_avg_last_ten_test_acc.append(df_results["avg_last_ten_test_acc"])
        five_avg_last_ten_test_f1.append(df_results["avg_last_ten_test_f1"])


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


######################################################################################################
if __name__ == '__main__':

    args = parse_args()
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    basicpath = os.path.dirname(father_path)
    # Logging setting
    if not args.debug:  # if not debug, no log.
        logger = get_logger(logging.INFO,args.debug,args=args,filename='logfile.log')
        __stderr__ = sys.stderr  #
        sys.stderr = open(create_logfile(args, 'error.log'), 'a')
        __stdout__ = sys.stdout
        sys.stdout = StreamToLogger(logger,logging.INFO)

    print(f"father_path = {father_path}")
    result_value = []

    if args.ucr==128:
        ucr=datasets.ucr_dataset_list()[args.from_ucr:args.end_ucr]
    else:
        ucr=['ArrowHead','CBF','FaceFour','MelbournePedestrian','OSULeaf','Plane','Symbols','Trace',
             'Epilepsy','NATOPS','EthanolConcentration', 'FaceDetection', 'FingerMovements']

    for dataset_name in ucr:
        args = parse_args()
        args.basicpath = basicpath
        args.dataset = dataset_name

        df_results = main(args, dataset_name)
        result_value.append(df_results)

        print(f'result_value = {result_value}')

        path = os.path.abspath(os.path.join(basicpath, 'statistic_results', args.outfile))
        pd.DataFrame(result_value).to_csv(path)


