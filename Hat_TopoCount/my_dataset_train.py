from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image
import glob


class CrowdDataset(Dataset):
    '''
    crowdDataset
    '''
    def __init__(self, img_root, gt_dmap_root, gt_dot_root, split_txt_filepath=None, phase='train', aug=0, normalize=True, fixed_size=-1, max_side=-1):
        '''
        img_root: the root path of images.
        gt_dmap_root: the root path of ground-truth custom (dilated) dot map.
        gt_dot_root: the root path of ground-truth dot map.
        phase: train or test
        split_txt_filepath: text file containing list of images to include in the dataset. If none, then use all jpg images in img_root
        '''
        self.img_root=img_root
        self.gt_dmap_root=gt_dmap_root
        self.gt_dot_root=gt_dot_root
        self.phase=phase
        self.split_txt_filepath = split_txt_filepath

        if(split_txt_filepath is None):
            self.img_names=[filename for filename in os.listdir(img_root) \
                               if os.path.isfile(os.path.join(img_root,filename))]
        else:
            img_list = np.loadtxt(split_txt_filepath, dtype=str)        
            self.img_names=[filename + '.jpg' for filename in img_list[:,0] \
                               if os.path.isfile(os.path.join(img_root,filename+ '.jpg'))]

        self.n_samples=len(self.img_names)

        self.aug=aug
        self.normalize = normalize;
        self.fixed_size = fixed_size
        self.max_side = max_side

        print('self.aug', self.aug)
        print('self.fixed_size', self.fixed_size)

    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.img_names[index]
        img=plt.imread(os.path.join(self.img_root,img_name))/255# convert from [0,255] to [0,1]
        
        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)
        img=img[:,:,0:3]

        gt_path = os.path.join(self.gt_dmap_root,img_name.replace('.jpg','.npy'));
        if(os.path.isfile(gt_path)):
            print('gt_path',gt_path)
            gt_dmap=np.load(gt_path)
        else:
            gt_dmap=np.zeros((img.shape[0], img.shape[1]))

        gtdot_path = os.path.join(self.gt_dot_root,img_name.replace('.jpg','_gt_dots.npy'));
        if(os.path.isfile(gtdot_path)):
            gt_dot=np.load(gtdot_path)
        else:
            gtdot_path = os.path.join(self.gt_dot_root,img_name.replace('.jpg','.npy'));
            if(os.path.isfile(gtdot_path)):
                gt_dot=np.load(gtdot_path)
            else:
                gt_dot=np.zeros((img.shape[0], img.shape[1]))

        
        if random.randint(0,1)==1 and self.phase=='train':
            img=img[:,::-1].copy() # horizontal flip
            gt_dmap=gt_dmap[:,::-1].copy() # horizontal flip
            gt_dot=gt_dot[:,::-1].copy() # horizontal flip
        
        if(self.phase=='train' and self.max_side > 0):
            h = img.shape[0]
            w = img.shape[1]
            h2 = h
            w2 = w
            crop = False
            if(h > self.max_side):
                h2 = self.max_side
                crop = True
            if(w > self.max_side):
                w2 = self.max_side
                crop = True
            if(crop):
                y=0
                x=0
                if(not (h2 ==h)):
                    y = np.random.randint(0, high = h-h2)
                if(not (w2 ==w)):
                    x = np.random.randint(0, high = w-w2)
                img = img[y:y+h2, x:x+w2, :]
                gt_dmap = gt_dmap[y:y+h2, x:x+w2]
                gt_dot = gt_dot[y:y+h2, x:x+w2]

        
        if ((self.aug > 0 and self.phase=='train')or (self.fixed_size > 0)):
            i = -1
            img_pil = Image.fromarray(img.astype(np.uint8)*255);
            if(self.fixed_size < 0):
                i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(img.shape[0]//4, img.shape[1]//4))
            elif(self.fixed_size < img.shape[0] or self.fixed_size < img.shape[1]):
                i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(min(self.fixed_size,img.shape[0]), min(self.fixed_size,img.shape[1])))
            #print('i, j, h, w',i, j, h, w)
            if(i >= 0):
                img = img[i:i+h, j:j+w, :]
                gt_dmap = gt_dmap[i:i+h, j:j+w]
                gt_dot = gt_dot[i:i+h, j:j+w]


        max_scale = 16
        if max_scale>1: # fix image and gt to match model.
            #ds_rows=int(img.shape[0]//max_scale)*max_scale
            #ds_cols=int(img.shape[1]//max_scale)*max_scale
            #img = img[:ds_rows, :ds_cols, :]
            #gt_dmap = gt_dmap[:ds_rows, :ds_cols]
            #gt_dot = gt_dot[:ds_rows, :ds_cols]
            ds_rows=int(img.shape[0]//max_scale)*max_scale
            ds_cols=int(img.shape[1]//max_scale)*max_scale
            pad_y1 = 0
            pad_y2 = 0
            pad_x1 = 0
            pad_x2 = 0
            if(ds_rows < img.shape[0]):
                pad_y1 = (max_scale - (img.shape[0] - ds_rows))//2
                pad_y2 = (max_scale - (img.shape[0] - ds_rows)) - pad_y1
            if(ds_cols < img.shape[1]):
                pad_x1 = (max_scale - (img.shape[1] - ds_cols))//2
                pad_x2 = (max_scale - (img.shape[1] - ds_cols)) - pad_x1
            img = np.pad(img, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(1,) )# padding constant differs by dataset based on bg color
            gt_dmap = np.pad(gt_dmap, ((pad_y1,pad_y2),(pad_x1,pad_x2)), 'constant', constant_values=(0,) )# padding constant differs by dataset based on bg color
            gt_dot = np.pad(gt_dot, ((pad_y1,pad_y2),(pad_x1,pad_x2)), 'constant', constant_values=(0,) )# padding constant differs by dataset based on bg color

        gt_dmap=gt_dmap[np.newaxis,:,:]
        gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
        gt_dot=gt_dot[np.newaxis,:,:]
        gt_dot_tensor=torch.tensor(gt_dot,dtype=torch.float)

        img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)
        img_tensor=torch.tensor(img,dtype=torch.float)
        if(self.normalize):
            img_tensor=transforms.functional.normalize(img_tensor,mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        return img_tensor,gt_dmap_tensor,gt_dot_tensor


