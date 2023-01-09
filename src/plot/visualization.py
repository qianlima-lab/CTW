import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
import torch
import tsaug
import time
import os
import argparse
import numpy as np
import sys
sys.path.append(os.path.dirname(sys.path[0]))

from src.models.MultiTaskClassification import MetaModel, NonLinClassifier
from src.models.model import CNNAE
from src.models.model import MetaModel_AE
from src.utils.saver import Saver
from src.utils.global_var import OUTPATH
from src.ucr_data.load_ucr_pre import load_ucr
from src.utils.utils import flip_label,create_synthetic_dataset,to_one_hot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='', help='data root')
parser.add_argument('--model', choices=['co_teaching', 'co_teaching_mloss', 'sigua',
                                        'co_teaching_ae_mloss_aug','single_ae_aug_after_sel', 'single_aug', 'single_sel', 'vanilla',
                                            'single_aug_after_sel', 'single_ae_sel', 'single_ae', 'single_ae_aug',
                                            'single_ae_aug_before_sel'])
parser.add_argument('--dataset', type=str, default='CBF', help='UCR datasets')
parser.add_argument('--datasets', type=str,nargs='+', default='CBF', help='UCR datasets')

parser.add_argument('--ni', type=float, default=0.5, help='label noise ratio')
parser.add_argument('--label_noise', type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                                   'number as str for time-dependent noise')
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
parser.add_argument('--aug',
                    choices=['GNoise', 'NoAug', 'Oversample', 'Convolve', 'Crop', 'Drift', 'TimeWarp', 'Mixup'],
                    default='NoAug')
parser.add_argument('--nbins', type=int, default=0, help='number of class')
parser.add_argument('--datestr', type=str, default='date')
parser.add_argument('--cuda_device', type=int, default=0, help='cuda number')
parser.add_argument('--sample_len', type=int,default=0)
parser.add_argument('--window', type=str,choices=['single','all'],default='all',
                    help='single_train/single_test: only plot training/test data; all: plot all data ')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def t_sne(xs, ys, y_clean, tsne=True, model=None, args=None,datestr='',sel_dict=None):
    ori_size = len(xs)
    aug_alpha = 1
    ori_alpha = 1
    # map_color = {0: 'r', 1: 'b',2:'c',3: 'm',4: 'g',5: 'k',6: 'y',7:'chocolate',
    #              8:'darkorange',9:'lawngreen',10:'cyan',11:'m',12:'deeppink'}
    map_color = {0: '#585EFF', 1: '#EE002B', 2: '#C791FF', 3: '#00C29F', 4: 'm', 5: 'k', 6: 'b', 7: 'chocolate',
                 8: 'darkorange', 9: 'lawngreen', 10: 'cyan', 11: 'm', 12: 'deeppink'}
    if args.label_noise == -1:
        label_noise = 'inst{}'.format(int(args.ni*100))
    elif args.label_noise == 0:
        label_noise = 'sym{}'.format(int(args.ni*100))
    else:
        label_noise = 'asym{}'.format(int(args.ni*100))

    datasetname = args.dataset
    if tsne:
        tsne = TSNE(n_components=2, random_state=args.seed)
    else:
        tsne = MDS(n_components=2, random_state=args.seed)

    xs = xs.to(device)
    # xs = torch.unsqueeze(xs, 1)

    model = model
    # classifier = model.classifier

    model.load_state_dict(
        torch.load('{}/src/model_save/{}/{}{}_{}_{}.pt'.format(args.basicpath,datasetname,args.model,args.aug,label_noise,datestr), map_location='cuda:{}'.format(int(args.cuda_device))))
    model.eval()
    features = model.encoder(xs).squeeze(-1)
    model_out = model.classifier(features).squeeze(-1)
    pred = torch.argmax(model_out, 1).int().cpu().numpy()

    if args.model in ['single_ae_aug_after_sel','single_ae_aug_before_sel','single_aug','single_ae_aug']:
        if sel_dict is None:
            sel_dict = np.load('{}/src/model_save/{}/{}{}_{}_{}_sel_dict.npy'.format(args.basicpath, datasetname, args.model, args.aug,
                                                                label_noise, datestr), allow_pickle=True).item()
        if args.aug in ['Mixup']:
            pass
        else:
            sel_ind = np.concatenate(sel_dict['sel_ind'],axis=0)
            x_sel = xs[sel_ind].cpu().numpy()
    else:
        x_sel=xs.cpu().numpy()


    plt.suptitle(args.model + '-' + args.aug, size=20)

    if args.aug=='NoAug':
        xs_out = tsne.fit_transform(xs.cpu().numpy().squeeze(-1), y_clean)
        feature_map = tsne.fit_transform(features.cpu().detach().numpy())
        f1 = plt.figure(figsize=(20, 10))
        f1.add_subplot(1, 5, 1)
        plt.title('Raw Clean', fontdict={'size': 10})
        plt.scatter(xs_out[:, 0], xs_out[:, 1], c=y_clean)

        f1.add_subplot(1, 5, 2)
        plt.title('Raw Observed', fontdict={'size': 10})
        plt.scatter(xs_out[:, 0], xs_out[:, 1], c=ys)

        f1.add_subplot(1, 5, 3)
        plt.title('EM with observed label', fontdict={'size': 10})
        plt.scatter(feature_map[:, 0], feature_map[:, 1], c=ys)

        f1.add_subplot(1, 5, 4)
        plt.title('EM with predicted label', fontdict={'size': 10})
        plt.scatter(feature_map[:, 0], feature_map[:, 1], c=pred)

        f1.add_subplot(1, 5, 5)
        plt.title('EM with clean label', fontdict={'size': 10})
        plt.scatter(feature_map[:, 0], feature_map[:, 1], c=y_clean)
    x_aug=None
    if args.aug == 'GNoise':
        x_aug = torch.from_numpy(
            tsaug.AddNoise(scale=0.015).augment(x_sel)).float().to(device)
    elif args.aug == 'Oversample':
        x_aug = torch.from_numpy(x_sel).to(device)
    elif args.aug == 'Convolve':
        x_aug = torch.from_numpy(
            tsaug.Convolve(window='flattop', size=10).augment(x_sel)).float().to(device)
    elif args.aug == 'Crop':
        x_aug = torch.from_numpy(
            tsaug.Crop(size=int(args.sample_len * (2 / 3)), resize=int(args.sample_len)).augment(
                x_sel)).float().to(
            device)
    elif args.aug == 'Drift':
        x_aug = torch.from_numpy(
            tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(
                x_sel)).float().to(
            device)
    elif args.aug == 'TimeWarp':
        x_aug = torch.from_numpy(
            tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                x_sel)).float().to(
            device)

    if args.aug !='NoAug':

        if args.aug == 'Mixup':
            xaug_features=torch.tensor([]).to(device)
            mix_xs=torch.tensor([]).to(device)
            mix_ys=np.array([]).astype(int)
            mix_yclean=np.array([]).astype(int)
            for i in range(len(sel_dict['lam'])):
                mix_h = sel_dict['lam'][i] * features[sel_dict['sel_ind'][i]] + (1 - sel_dict['lam'][i]) * features[sel_dict['mix_ind'][i]]
                mix_x = sel_dict['lam'][i] * xs[sel_dict['sel_ind'][i]] + (1 - sel_dict['lam'][i]) * xs[sel_dict['mix_ind'][i]]
                xaug_features=torch.cat([xaug_features,mix_h],dim=0)
                mix_xs=torch.cat([mix_xs,mix_x],dim=0)
                ta, tb = ys[sel_dict['sel_ind'][i]], ys[sel_dict['mix_ind'][i]]
                ys_aug = np.argmax(sel_dict['lam'][i] * to_one_hot(args.nbins,ta) + (1 - sel_dict['lam'][i])* to_one_hot(args.nbins,tb),axis=1)
                mix_ys=np.concatenate((mix_ys,ys_aug))
                y_clean_aug = np.argmax(sel_dict['lam'][i] * to_one_hot(args.nbins,y_clean[sel_dict['sel_ind'][i]]) + (1 - sel_dict['lam'][i]) *
                               to_one_hot(args.nbins,y_clean[sel_dict['mix_ind'][i]]),1)
                mix_yclean=np.concatenate([mix_yclean,y_clean_aug])
            ys=np.concatenate((ys,mix_ys))
            y_clean=np.concatenate((y_clean,mix_yclean))
            xs = np.concatenate((xs.cpu().numpy(),mix_xs.cpu().numpy()))
            features=np.concatenate((features.cpu().detach().numpy(),xaug_features.squeeze(-1).cpu().detach().numpy()))

            # xs_out = tsne.fit_transform(xs.cpu().detach().numpy().squeeze(-1), y_clean[:ori_size])
            #
            # f1.add_subplot(1, 5, 1)
            # plt.title('Raw Clean', fontdict={'size': 10})
            # plt.scatter(xs_out[:, 0], xs_out[:, 1], c=y_clean[:ori_size])
            #
            # f1.add_subplot(1, 5, 2)
            # plt.title('Raw Observed', fontdict={'size': 10})
            # plt.scatter(xs_out[:, 0], xs_out[:, 1], c=ys[:ori_size])
        else:
            xs=np.concatenate((xs.cpu().numpy(),x_aug.cpu().numpy()))
            y_clean=np.concatenate((y_clean,y_clean[sel_ind]))
            ys=np.concatenate((ys,ys[sel_ind]))
            xaug_features = model.encoder(x_aug).squeeze(-1)
            features = np.concatenate((features.cpu().detach().numpy(),xaug_features.cpu().detach().numpy()))

        xs_out = tsne.fit_transform(xs.squeeze(-1), y_clean)

        if args.window == 'single':
            feature_map = tsne.fit_transform(features, y_clean)
            f1 = plt.figure(figsize=(1.6,1.6),dpi=300)
            f1.add_subplot(1, 1, 1)
            plt.scatter(feature_map[int(ori_size/5+1):ori_size, 0], feature_map[int(ori_size/5+1):ori_size, 1],
                        c=list(map(lambda x: map_color[x], y_clean[int(ori_size/5+1):ori_size])), alpha=ori_alpha,s=3)
            plt.scatter(feature_map[ori_size:, 0], feature_map[ori_size:, 1],
                        c=list(map(lambda x: map_color[x], y_clean[ori_size:])), alpha=aug_alpha, s=3)

            feature_map = tsne.fit_transform(features, y_clean)
            plt.xticks(fontsize=4)
            plt.yticks(fontsize=4)
            # plt.tight_layout()
            f2 = plt.figure(figsize=(1.6, 1.6),dpi=350)
            f2.add_subplot(1, 1, 1)
            plt.scatter(feature_map[:int(ori_size/5+1), 0], feature_map[:int(ori_size/5+1), 1],
                        c=list(map(lambda x: map_color[x], y_clean[:int(ori_size/5+1)])), alpha=ori_alpha,s=3)
            plt.xticks(fontsize=4)
            plt.yticks(fontsize=4)
            # plt.tight_layout()

            # plt.scatter(feature_map[ori_size:, 0], feature_map[ori_size:, 1],
            #             c=list(map(lambda x: map_color[x], y_clean[ori_size:])), alpha=aug_alpha, marker='x')
        else:
            f1 = plt.figure(figsize=(10, 5))
            f1.add_subplot(1, 5, 1)
            plt.title('Raw Clean', fontdict={'size': 10})
            plt.scatter(xs_out[:ori_size, 0], xs_out[:ori_size, 1],
                        c=list(map(lambda x: map_color[x], y_clean[:ori_size])), alpha=ori_alpha)
            plt.scatter(xs_out[ori_size:, 0], xs_out[ori_size:, 1],
                        c=list(map(lambda x: map_color[x], y_clean[ori_size:])), alpha=aug_alpha, marker='x')

            f1.add_subplot(1, 5, 2)
            plt.title('Raw Observed', fontdict={'size': 10})
            plt.scatter(xs_out[:ori_size, 0], xs_out[:ori_size, 1], c=list(map(lambda x: map_color[x], ys[:ori_size])),
                        alpha=ori_alpha)
            plt.scatter(xs_out[ori_size:, 0], xs_out[ori_size:, 1], c=list(map(lambda x: map_color[x], ys[ori_size:])),
                        alpha=aug_alpha, marker='x')

            feature_map = tsne.fit_transform(features, y_clean)

            f1.add_subplot(1, 5, 3)
            plt.title('EM with observed label', fontdict={'size': 10})
            plt.scatter(feature_map[:ori_size, 0], feature_map[:ori_size, 1],
                        c=list(map(lambda x: map_color[x], ys[:ori_size])), alpha=ori_alpha)
            plt.scatter(feature_map[ori_size:, 0], feature_map[ori_size:, 1],
                        c=list(map(lambda x: map_color[x], ys[ori_size:])), alpha=aug_alpha, marker='x')

            f1.add_subplot(1, 5, 4)
            plt.title('EM with predicted label', fontdict={'size': 10})
            plt.scatter(feature_map[:ori_size, 0], feature_map[:ori_size, 1], c=list(map(lambda x: map_color[x], pred)),
                        alpha=ori_alpha)
            plt.scatter(feature_map[ori_size:, 0], feature_map[ori_size:, 1],
                        c=list(map(lambda x: map_color[x], pred[np.concatenate(sel_dict['sel_ind'])])), alpha=aug_alpha,
                        marker='x')

            f1.add_subplot(1, 5, 5)
            plt.title('EM with clean label', fontdict={'size': 10})
            plt.scatter(feature_map[:ori_size, 0], feature_map[:ori_size, 1],
                        c=list(map(lambda x: map_color[x], y_clean[:ori_size])), alpha=ori_alpha)
            plt.scatter(feature_map[ori_size:, 0], feature_map[ori_size:, 1],
                        c=list(map(lambda x: map_color[x], y_clean[ori_size:])), alpha=aug_alpha, marker='x')

    plt.tight_layout()
    savefigpath = os.path.join(args.basicpath, 'src', '../visualization', args.dataset)
    os.makedirs(savefigpath, exist_ok=True)

    datestr = time.strftime(('%Y%m%d'))

    filename='{}{}_{}_{}_{}'.format(os.path.join(savefigpath,args.model),args.aug,label_noise,args.dataset,datestr)
    i = 0
    while os.path.exists('{}_{:d}.png'.format(filename, i)):
        i += 1
    if args.window=='single':
        f1.savefig('{}_trainset{:d}.png'.format(filename, i))
        f2.savefig('{}_testset{:d}.png'.format(filename, i))
    else:
        f1.savefig('{}_{:d}.png'.format(filename, i))

    plt.clf()

def t_sne_during_train(xs, ys, y_clean, tsne=True, model=None, args=None,datestr='',sel_dict=None,epoch=None):
    ori_size = len(xs)
    aug_alpha = 1.
    ori_alpha = 0.2
    map_color = {0: 'r', 1: 'b',2:'c',3: 'm',4: 'g',5: 'k',6: 'y',7:'chocolate',
                 8:'darkorange',9:'lawngreen',10:'cyan',11:'m',12:'deeppink'}

    if args.label_noise == -1:
        label_noise = 'inst{}'.format(int(args.ni*100))
    elif args.label_noise == 0:
        label_noise = 'sym{}'.format(int(args.ni*100))
    else:
        label_noise = 'asym{}'.format(int(args.ni*100))

    datasetname = args.dataset
    if tsne:
        tsne = TSNE(n_components=2, random_state=args.seed)
    else:
        tsne = MDS(n_components=2, random_state=args.seed)

    xs = xs.to(device)
    # xs = torch.unsqueeze(xs, 1)

    model = model
    # classifier = model.classifier

    model.eval()
    features = model.encoder(xs).squeeze(-1)
    model_out = model.classifier(features).squeeze(-1)
    pred = torch.argmax(model_out, 1).int().cpu().numpy()

    if args.model in ['single_ae_aug_after_sel','single_ae_aug_before_sel','single_aug','single_ae_aug']:
        if sel_dict is None:
            sel_dict = np.load('{}/src/model_save/{}/{}{}_{}_{}_sel_dict.npy'.format(args.basicpath, datasetname, args.model, args.aug,
                                                                label_noise, datestr), allow_pickle=True).item()
        if args.aug in ['Mixup']:
            pass
        else:
            sel_ind = np.concatenate(sel_dict['sel_ind'],axis=0)
            x_sel = xs[sel_ind].cpu().numpy()
    else:
        x_sel=xs.cpu().numpy()

    f1 = plt.figure(figsize=(10, 20))
    plt.suptitle(args.model + '-' + args.aug, size=20)

    if args.aug=='NoAug':
        xs_out = tsne.fit_transform(xs.cpu().numpy().squeeze(-1), y_clean)
        feature_map = tsne.fit_transform(features.cpu().detach().numpy())

        f1.add_subplot(2, 1, 1)
        plt.title('EM with predicted label', fontdict={'size': 10})
        plt.scatter(feature_map[:, 0], feature_map[:, 1], c=pred)

        f1.add_subplot(2, 1, 2)
        plt.title('EM with clean label', fontdict={'size': 10})
        plt.scatter(feature_map[:, 0], feature_map[:, 1], c=y_clean)
    x_aug=None
    if args.aug == 'GNoise':
        x_aug = torch.from_numpy(
            tsaug.AddNoise(scale=0.015).augment(x_sel)).float().to(device)
    elif args.aug == 'Oversample':
        x_aug = torch.from_numpy(x_sel).to(device)
    elif args.aug == 'Convolve':
        x_aug = torch.from_numpy(
            tsaug.Convolve(window='flattop', size=10).augment(x_sel)).float().to(device)
    elif args.aug == 'Crop':
        x_aug = torch.from_numpy(
            tsaug.Crop(size=int(args.sample_len * (2 / 3)), resize=int(args.sample_len)).augment(
                x_sel)).float().to(
            device)
    elif args.aug == 'Drift':
        x_aug = torch.from_numpy(
            tsaug.Drift(max_drift=0.2, n_drift_points=5).augment(
                x_sel)).float().to(
            device)
    elif args.aug == 'TimeWarp':
        x_aug = torch.from_numpy(
            tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3).augment(
                x_sel)).float().to(
            device)

    if args.aug !='NoAug':

        if args.aug == 'Mixup':
            xaug_features=torch.tensor([]).to(device)
            mix_xs=torch.tensor([]).to(device)
            mix_ys=np.array([]).astype(int)
            mix_yclean=np.array([]).astype(int)
            for i in range(len(sel_dict['lam'])):
                mix_h = sel_dict['lam'][i] * features[sel_dict['sel_ind'][i]] + (1 - sel_dict['lam'][i]) * features[sel_dict['mix_ind'][i]]
                mix_x = sel_dict['lam'][i] * xs[sel_dict['sel_ind'][i]] + (1 - sel_dict['lam'][i]) * xs[sel_dict['mix_ind'][i]]
                xaug_features=torch.cat([xaug_features,mix_h],dim=0)
                mix_xs=torch.cat([mix_xs,mix_x],dim=0)
                ta, tb = ys[sel_dict['sel_ind'][i]], ys[sel_dict['mix_ind'][i]]
                ys_aug = np.argmax(sel_dict['lam'][i] * to_one_hot(args.nbins,ta) + (1 - sel_dict['lam'][i])* to_one_hot(args.nbins,tb),axis=1)
                mix_ys=np.concatenate((mix_ys,ys_aug))
                y_clean_aug = np.argmax(sel_dict['lam'][i] * to_one_hot(args.nbins,y_clean[sel_dict['sel_ind'][i]]) + (1 - sel_dict['lam'][i]) *
                               to_one_hot(args.nbins,y_clean[sel_dict['mix_ind'][i]]),1)
                mix_yclean=np.concatenate([mix_yclean,y_clean_aug])
            ys=np.concatenate((ys,mix_ys))
            y_clean=np.concatenate((y_clean,mix_yclean))
            xs = np.concatenate((xs.cpu().numpy(),mix_xs.cpu().numpy()))
            features=np.concatenate((features.cpu().detach().numpy(),xaug_features.squeeze(-1).cpu().detach().numpy()))

        else:
            xs=np.concatenate((xs.cpu().numpy(),x_aug.cpu().numpy()))
            y_clean=np.concatenate((y_clean,y_clean[sel_ind]))
            ys=np.concatenate((ys,ys[sel_ind]))
            xaug_features = model.encoder(x_aug).squeeze(-1)
            features = np.concatenate((features.cpu().detach().numpy(),xaug_features.cpu().detach().numpy()))

        feature_map = tsne.fit_transform(features,y_clean)

        f1.add_subplot(2, 1, 1)
        plt.title('EM with predicted label', fontdict={'size': 10})
        plt.scatter(feature_map[:ori_size, 0], feature_map[:ori_size, 1], c=list(map(lambda x: map_color[x],pred)),alpha=ori_alpha)
        plt.scatter(feature_map[ori_size:, 0], feature_map[ori_size:, 1], c=list(map(lambda x: map_color[x],pred[np.concatenate(sel_dict['sel_ind'])])), alpha=aug_alpha,marker='x')

        f1.add_subplot(2, 1, 2)
        plt.title('EM with clean label', fontdict={'size': 10})
        plt.scatter(feature_map[:ori_size, 0], feature_map[:ori_size, 1], c=list(map(lambda x: map_color[x],y_clean[:ori_size])),alpha=ori_alpha)
        plt.scatter(feature_map[ori_size:, 0], feature_map[ori_size:, 1], c=list(map(lambda x: map_color[x],y_clean[ori_size:])),alpha=aug_alpha,marker='x')

    plt.tight_layout()
    savefigpath = os.path.join(args.basicpath, 'src', '../visualization', args.dataset)
    os.makedirs(savefigpath, exist_ok=True)

    datestr = time.strftime(('%Y%m%d'))

    filename='{}{}_{}_{}_{}_epoch{}'.format(os.path.join(savefigpath,args.model),args.aug,label_noise,args.dataset,datestr,epoch)
    i = 0
    while os.path.exists('{}_{:d}.png'.format(filename, i)):
        i += 1

    f1.savefig('{}_{:d}.png'.format(filename, i))

    plt.clf()

def main(args):
    if args.dataset=='synthesis':
        X, Y=create_synthetic_dataset(ts_n=800)
    else:
        X, Y = load_ucr(args.dataset)
    args.sample_len = X.shape[1]
    Y_clean=Y.copy()
    classes = len(np.unique(Y))
    args.nbins = classes
    Y_noise, _ = flip_label(X, Y, args.ni, args)
    classifier1 = NonLinClassifier(args.embedding_size, args.nbins, d_hidd=args.classifier_dim, dropout=args.dropout,
                                   norm=args.normalization)

    model = CNNAE(input_size=X.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                  seq_len=X.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                  padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)
    if args.model in ['single_ae_aug_after_sel', 'single_ae', 'single_ae_sel', 'single_ae_aug',
                      'single_ae_aug_before_sel']:
        model = MetaModel_AE(ae=model, classifier=classifier1, name='CNN').to(device)
    elif args.model in ['single_aug_after_sel', 'single_aug', 'single_sel', 'vanilla']:
        model = MetaModel(ae=model, classifier=classifier1, name='CNN').to(device)

    t_sne(torch.from_numpy(X).float(), Y_noise, Y_clean, model=model, tsne=True, args=args,datestr=args.datestr)

if __name__ == '__main__':
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

    args = parser.parse_args()
    args=parser.parse_args('--model single_ae_aug_before_sel --label_noise 0 --aug TimeWarp --dataset Trace --datestr 20220802 \
--embedding_size 32 --ni 0.3 --cuda_device 0 --window single'.split(' '))

    # args.dataset='synthesis'
    augs=['GNoise','Drift','Crop','TimeWarp','Convolve','Oversample','Mixup']
    # augs=[args.aug]
    if device == torch.device('cuda'):
        torch.cuda.set_device(args.cuda_device)
    args.basicpath = os.path.dirname(father_path)

    for aug_ in augs:
        args.aug=aug_
        main(args)
        print("model: {} ; dataset: {} ; t-sne in {}".format(args.model, args.dataset,
                                                             os.path.join(args.basicpath, 'src', '../visualization', args.dataset, args.model)))


