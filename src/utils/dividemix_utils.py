from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter
import tsaug

            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class TS_dataset(Dataset):
    def __init__(self, dataset, r, transform, mode,train_dataset,test_dataset, noise_file='', pred=[], probability=[], log=''):
        
        self.r = r # noise ratio
        self.transform = transform
        self.train_data=train_dataset
        self.test_data=test_dataset
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:3,6:6,8:8} # class transition for asymmetric noise
     
        if self.mode=='test':
            _,self.test_label = test_dataset.tensors
        else:
            train_data,noise_label,_,train_label=train_dataset.tensors
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
            else:                   
                if self.mode == "labeled":
                    pred_idx = pred.nonzero()[0]
                    self.probability = [probability[i] for i in pred_idx]   
                    
                    clean = (np.array(noise_label)==np.array(train_label))                                                       
                    auc_meter = AUCMeter()
                    auc_meter.reset()
                    auc_meter.add(probability,clean)        
                    auc,_,_ = auc_meter.value()               
                    # log.write('Numer of labeled samples:%d   AUC:%.3f\n'%(pred.sum(),auc))
                    # log.flush()
                    
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]                                               
                
                self.train_data = train_data[pred_idx]
                # self.noise_label = [noise_label[i] for i in pred_idx]
                self.noise_label = noise_label[pred_idx]
                print("%s data has a size of %d"%(self.mode,len(self.noise_label)))            
                
    def __getitem__(self, index):
        if self.mode=='labeled':
            img, target, prob = self.train_data[index], self.noise_label[index], self.probability[index]
            # img = Image.fromarray(img)
            # img1=self.transform(img)
            # img2=self.transform(img)
            # img1 = torch.from_numpy(self.transform.augment(img.unsqueeze(0).cpu().numpy()).squeeze(0)).cuda()
            # img2 = torch.from_numpy(self.transform.augment(img.unsqueeze(0).cpu().numpy()).squeeze(0)).cuda()
            img1=img.clone()
            img2=img.clone()
            return img1, img2, target, prob
        elif self.mode=='unlabeled':
            img = self.train_data[index]
            # img = Image.fromarray(img)
            # img1 = torch.from_numpy(self.transform.augment(img.unsqueeze(0).cpu().numpy()).squeeze(0)).cuda()
            # img2 = torch.from_numpy(self.transform.augment(img.unsqueeze(0).cpu().numpy()).squeeze(0)).cuda()
            img1=img.clone()
            img2=img.clone()
            return img1, img2
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            # img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            # img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class dividemix_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, train_dataset,test_dataset,sample_len):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_len = sample_len
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.transform_train = (tsaug.TimeWarp(n_speed_change=5, max_speed_ratio=3))
        self.transform_test = lambda x:x

    def run(self,mode,pred=[],prob=[]):
        if mode=='warmup':
            all_dataset = self.train_dataset
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='train':
            labeled_dataset = TS_dataset(dataset=self.dataset, r=self.r,
                                            transform=self.transform_train, mode="labeled",
                                             pred=pred, probability=prob,train_dataset=self.train_dataset,
                                            test_dataset=self.test_dataset)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)   
            
            unlabeled_dataset = TS_dataset(dataset=self.dataset,
                                              r=self.r, transform=self.transform_train,
                                              mode="unlabeled", pred=pred,train_dataset=self.train_dataset,
                                              test_dataset=self.test_dataset)
            try:
                unlabeled_trainloader = DataLoader(
                    dataset=unlabeled_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers)
            except:
                unlabeled_trainloader=None

            return labeled_trainloader, unlabeled_trainloader
        
        elif mode=='test':
            test_dataset = TS_dataset(dataset=self.dataset, r=self.r,
                                         transform=self.transform_test, mode='test',train_dataset=self.train_dataset,
                                         test_dataset=self.test_dataset)
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = TS_dataset(dataset=self.dataset, r=self.r,
                                        transform=self.transform_test, mode='all',train_dataset=self.train_dataset,
                                         test_dataset=self.test_dataset)
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader