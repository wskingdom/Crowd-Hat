#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.cm as CM

import numpy as np
import time
import torch
import torch.nn as nn
import os
import random
from tqdm import tqdm as tqdm 
import sys;
import math
import skimage.io as io
from scipy import ndimage
from scipy.misc import imresize
from skimage.measure import label
from skimage import filters

from unet_vgg4_cc import UnetVggCC
from my_dataset_train import CrowdDataset
from TDFMain_pytorch import *


'''
 Models checkpoints and training log file are saved in checkpoints_save_path
 train epochs stats are saved in checkpoints_save_path: epochs_train_loss_dice_list.npy, epochs_train_loss_pers_list.npy, epochs_train_mae_list.npy, epochs_train_rmse_list.npy, epochs_train_2rmse_plus_mae_list.npy
 test epochs stats are saved in checkpoints_save_path: epochs_test_loss_dice_list.npy, epochs_test_mae_list.npy, epochs_test_rmse_list.npy, epochs_test_2rmse_plus_mae_list.npy

 Below are some default configurations for the datasets: ShanghaTech Part A, ShanghaTech Part B, UCF-QNRF, JHU++, NWPU-Crowd.
 Uncomment the approporiate configuration

The configurations include:
    model_param_path: If just starting to train then set to None. If continuing from a pretrained crowd counting model then set the path of the pretrained model here.
    checkpoints_save_path: checkpoints/ models save directory. In addition to sample train/test and training log file.
    root: dataset root directory.
    train_image_root: training dataset images directory.
    train_dmap_root: training dataset ground truth topological map directory.
    train_dots_root: training dataset ground truth dot map directory.
    train_split_txt_filepath: filepath containing image filenames in the training split. If set to None, will use all the images in train_image_root.
    test_image_root: test/validation dataset images directory.
    test_dmap_root: test/validation dataset ground truth topological map directory.
    test_dots_root: test/validation dataset ground truth dot map directory.
    test_split_txt_filepath: filepath containing image filenames in the test/validation split. If set to None, will use all the images in test_image_root.

    topo_size: tiling patch size for persistence loss
    start_epoch: start epoch numbering. useful if stop and continue in same directory
    lamda_pers: weight for persistence loss
    lamda_dice: weight for dice loss
    epoch_start_pers_loss: default epoch to start adding persistence loss. Idealy chosen manually when the model starts to output reasonable predictions from which topology can be inferred
    train_patch_size: size of image patch to use to train. -1 means whole image. otherwise random crops of size train_patch_size x train_patch_size are used
    test_patch_size: size of image patch to use to test. -1 means whole image. otherwise random crops of size test_patch_size x test_patch_size are used. If get cuda error, change to 1024 and then run a separate evaluation on the trained epochs to select optimized model.

'''
###################################################################################
# Configuration for ShanghaiTech Part A
###################################################################################


if __name__=="__main__":

    # Below are some default configurations for the datasets: ShanghaTech Part A, ShanghaTech Part B, UCF-QNRF, JHU++, NWPU-Crowd.
    # Uncomment the approporiate configuration
    ###################################################################################
    # Configuration for ShanghaiTech Part A
    ###################################################################################    
    '''
    model_param_path = None; 
    #model_param_path = '../checkpoints/sh_parta_custom_topo1_patch50/epoch_4.pth';      
    checkpoints_save_path = './checkpoints/sh_parta_custom_topo1_patch50';

    root = './datasets/ShanghaiTech/'
    train_image_root = os.path.join(root,'part_A/train_data','images') 
    train_dmap_root = os.path.join(root,'part_A/train_data','gt_map_custom2')
    train_dots_root = os.path.join(root,'part_A/train_data','ground-truth_dots')
    train_split_txt_filepath = None
    test_image_root = os.path.join(root,'part_A/test_data','images') 
    test_dmap_root = os.path.join(root,'part_A/test_data','gt_map_custom2')
    test_dots_root = os.path.join(root,'part_A/test_data','ground-truth_dots')
    test_split_txt_filepath = None

    topo_size         = 50; # tiling patch size for persistence loss
    start_epoch = 0         # start epoch numbering. useful if stop and continue in same directory
    lamda_pers            = 1; # weight for persistence loss
    lamda_dice            = 1; # weight for dice loss
    epoch_start_pers_loss = 30 # default epoch to start adding persistence loss. Idealy chosen manually when the model starts to output reasonable predictions from which topology can be inferred
    train_patch_size = -1 # size of image patch to use to train. -1 means whole image. otherwise random crops of size train_patch_size x train_patch_size are used
    test_patch_size = -1 # size of image patch to use to test. -1 means whole image. otherwise random crops of size test_patch_size x test_patch_size are used. If get cuda error, change to 1024 and then run a separate evaluation on the trained epochs to select optimized model.
    '''

    ####################################################################################
    ## Configuration for ShanghaiTech Part B
    ####################################################################################
    #'''
    model_param_path = None; 
    #model_param_path = './checkpoints/sh_partb_custom_topo1_patch50/epoch_4.pth';      
    checkpoints_save_path = './checkpoints/sh_partb_custom_topo1_patch50';

    root = './datasets/ShanghaiTech/'
    train_image_root = os.path.join(root,'part_B/train_data','images') 
    train_dmap_root = os.path.join(root,'part_B/train_data','gt_map_custom2')
    train_dots_root = os.path.join(root,'part_B/train_data','ground-truth_dots')
    train_split_txt_filepath = None
    test_image_root = os.path.join(root,'part_B/test_data','images') 
    test_dmap_root = os.path.join(root,'part_B/test_data','gt_map_custom2')
    test_dots_root = os.path.join(root,'part_B/test_data','ground-truth_dots')
    test_split_txt_filepath = None

    topo_size         = 50; # tiling patch size for persistence loss
    start_epoch = 0         # start epoch numbering. useful if stop and continue in same directory
    lamda_pers            = 1; # weight for persistence loss
    lamda_dice            = 1; # weight for dice loss
    epoch_start_pers_loss = 3 # default epoch to start adding persistence loss. Idealy chosen manually when the model starts to output reasonable predictions from which topology can be inferred
    train_patch_size = -1 # size of image patch to use to train. -1 means whole image. otherwise random crops of size train_patch_size x train_patch_size are used
    test_patch_size = -1 # size of image patch to use to test. -1 means whole image. otherwise random crops of size test_patch_size x test_patch_size are used. If get cuda error, change to 1024 and then run a separate evaluation on the trained epochs to select optimized model.
    #'''

    ####################################################################################
    ## Configuration for UCF-QNRF
    ####################################################################################
    '''
    model_param_path = None; 
    #model_param_path = './checkpoints/qnrf_custom_topo1_patch100/epoch_4.pth';      
    checkpoints_save_path = './checkpoints/qnrf_custom_topo1_patch100';

    root = './datasets/UCF-QNRF_ECCV18/UCF-QNRF_ECCV18'
    train_image_root = os.path.join(root,'Train','img_scalelong2048')
    train_dmap_root = os.path.join(root,'Train','gt_map_custom2_scalelong2048')    
    train_dots_root = os.path.join(root,'Train','ground-truth_dots_scalelong2048')    
    train_split_txt_filepath = None
    test_image_root = os.path.join(root,'Test','img_scaleshort2048')
    test_dmap_root = os.path.join(root,'Test','gt_map_custom2_scaleshort2048')
    test_dots_root = os.path.join(root,'Test','ground-truth_dots_scaleshort2048')
    test_split_txt_filepath = None

    topo_size         = 100; # tiling patch size for persistence loss
    start_epoch = 0         # start epoch numbering. useful if stop and continue in same directory
    lamda_pers            = 1; # weight for persistence loss
    lamda_dice            = 1; # weight for dice loss
    epoch_start_pers_loss = 30 # default epoch to start adding persistence loss. Idealy chosen manually when the model starts to output reasonable predictions from which topology can be inferred
    train_patch_size = 1024 # size of image patch to use to train. -1 means whole image. otherwise random crops of size train_patch_size x train_patch_size are used
    test_patch_size = -1 # size of image patch to use to test. -1 means whole image. otherwise random crops of size test_patch_size x test_patch_size are used. If get cuda error, change to 1024 and then run a separate evaluation on the trained epochs to select optimized model.
    '''

    ####################################################################################
    ## Configuration for JHU++
    ####################################################################################
    '''
    model_param_path = None; 
    #model_param_path = './checkpoints/jhu_custom_topo1_patch100/epoch_4.pth';      
    checkpoints_save_path = './checkpoints/jhu_custom_topo1_patch100';

    root = './datasets/jhu/jhu_crowd_v2.0'
    train_image_root = os.path.join(root, 'train','images')
    train_dmap_root = os.path.join(root, 'train','gt_map_custom2_boxes')    
    train_dots_root = os.path.join(root, 'train','ground-truth_dots')    
    train_split_txt_filepath = None
    test_image_root = os.path.join(root, 'val','images')
    test_dmap_root = os.path.join(root, 'val','gt_map_custom2_boxes')    
    test_dots_root = os.path.join(root, 'val','ground-truth_dots')    
    test_split_txt_filepath = None

    topo_size         = 100; # tiling patch size for persistence loss
    start_epoch = 0         # start epoch numbering. useful if stop and continue in same directory
    lamda_pers            = 1; # weight for persistence loss
    lamda_dice            = 1; # weight for dice loss
    epoch_start_pers_loss = 30 # default epoch to start adding persistence loss. Idealy chosen manually when the model starts to output reasonable predictions from which topology can be inferred
    train_patch_size = 1024 # size of image patch to use to train. -1 means whole image. otherwise random crops of size train_patch_size x train_patch_size are used
    test_patch_size = -1 # size of image patch to use to test. -1 means whole image. otherwise random crops of size test_patch_size x test_patch_size are used. If get cuda error, change to 1024 and then run a separate evaluation on the trained epochs to select optimized model.
    '''

    ####################################################################################
    ## Configuration for NWPU-Crowd
    ####################################################################################
    '''
    model_param_path = None; 
    #model_param_path = './checkpoints/nwpu_custom_topo1_patch100/epoch_4.pth';      
    checkpoints_save_path = './checkpoints/nwpu_custom_topo1_patch100';

    root = './datasets/nwpu-crowd'
    train_image_root = os.path.join(root,'images')
    train_dmap_root = os.path.join(root,'gt_map_custom2_boxes')    
    train_dots_root = os.path.join(root,'ground-truth_dots')    
    train_split_txt_filepath = os.path.join(root,'train.txt')    
    test_image_root = os.path.join(root,'images')
    test_dmap_root = os.path.join(root,'gt_map_custom2_boxes')    
    test_dots_root = os.path.join(root,'ground-truth_dots')    
    test_split_txt_filepath = os.path.join(root,'val.txt')    

    topo_size         = 100; # tiling patch size for persistence loss
    start_epoch = 0         # start epoch numbering. useful if stop and continue in same directory
    lamda_pers            = 1; # weight for persistence loss
    lamda_dice            = 1; # weight for dice loss
    epoch_start_pers_loss = 30 # default epoch to start adding persistence loss. Idealy chosen manually when the model starts to output reasonable predictions from which topology can be inferred
    train_patch_size = 1024 # size of image patch to use to train. -1 means whole image. otherwise random crops of size train_patch_size x train_patch_size are used
    test_patch_size = -1 # size of image patch to use to test. -1 means whole image. otherwise random crops of size test_patch_size x test_patch_size are used. If get cuda error, change to 1024 and then run a separate evaluation on the trained epochs to select optimized model.
    '''

    ###################################################################################

    gt_multiplier = 1    
    gpu_or_cpu='cuda' # use cuda or cpu
    lr                = 0.00005 
    batch_size        = 1
    #momentum          = 0.95
    epochs            = 100
    seed              = time.time()

    dropout_keep_prob = 1.0
    initial_pad = 126
    interpolate = 'False'
    conv_init = 'he'
    n_classes = 1
    n_channels = 1
    
    sub_patch_border_width = 5
    padwidth = 3;
    mm=1
    thresh_low=0.4
    thresh_high=0.5

    device=torch.device(gpu_or_cpu)
    torch.cuda.manual_seed(seed)
    model=UnetVggCC(kwargs={'dropout_keep_prob':dropout_keep_prob, 'initial_pad':initial_pad, 'interpolate':interpolate, 'conv_init':conv_init, 'n_classes':n_classes, 'n_channels':n_channels})
    if(not (model_param_path is None)):
        model.load_state_dict(torch.load(model_param_path), strict=False);
        print('model loaded')
    model.to(device)
    criterion_sig = nn.Sigmoid() # initialize sigmoid layer
    criterion_bce = nn.BCEWithLogitsLoss() # initialize loss function
    optimizer=torch.optim.Adam(model.parameters(),lr) 
    train_dataset=CrowdDataset(train_image_root,train_dmap_root, train_dots_root, split_txt_filepath=train_split_txt_filepath,phase='train', normalize=False, aug=0, fixed_size=train_patch_size)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True)
    test_dataset=CrowdDataset(test_image_root,test_dmap_root,test_dots_root, split_txt_filepath=test_split_txt_filepath, phase='test', normalize=False, aug=0, fixed_size=test_patch_size)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
    
    if not os.path.exists(checkpoints_save_path):
        os.mkdir(checkpoints_save_path)

    log_file = open(os.path.join(checkpoints_save_path, 'log_file.txt'), 'a+')

    if not os.path.exists(checkpoints_save_path):
        os.mkdir(checkpoints_save_path)
    min_mae=10000
    min_rmse=10000
    min_loss=10000
    min_rmse_mae=10000
    min_epoch_mae=0
    min_epoch_rmse=0
    min_epoch_loss=0
    min_epoch_rmse_mae=0
    #train_loss_list=[]
    train_loss_dice_list=[]
    train_loss_pers_list=[]
    train_mae_list=[]
    train_rmse_list=[]
    train_rmse_mae_list=[]
    test_loss_dice_list=[]
    test_mae_list=[]
    test_rmse_list=[]
    test_rmse_mae_list=[]
    for epoch in range(start_epoch,epochs):
        # training phase
        model.train()
        if os.path.isfile(os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+".pth")):
            continue;
        print('epoch=',epoch) ;
        log_file.write("epoch= {} \n".format(epoch))
        log_file.flush()
        sys.stdout.flush();
        epoch_loss_dice=0
        epoch_loss_pers=0
        mae=0;
        rmse=0
        for i,(img,gt_dmap, gt_dots) in enumerate(tqdm(train_loader)):
            img=img.to(device)
            gt_dmap = gt_dmap > 0
            gt_dmap = gt_dmap.type(torch.FloatTensor)
            gt_dmap=gt_dmap.to(device)
            # forward propagation        
            et_dmap=model(img)[:,:,2:-2,2:-2]
            print('et_dmap.min()', et_dmap.min())
            print('et_dmap.max()', et_dmap.max())

            loss_pers = torch.tensor(0)
            if(lamda_pers > 0 and epoch >= epoch_start_pers_loss):
                n_fix = 0
                n_remove = 0
                topo_cp_weight_map = np.zeros(et_dmap.shape);
                topo_cp_weight_map_vis_b_fix = np.zeros(et_dmap.shape);
                topo_cp_weight_map_vis_d_fix = np.zeros(et_dmap.shape);
                topo_cp_weight_map_vis_b_rem = np.zeros(et_dmap.shape);
                topo_cp_weight_map_vis_d_rem = np.zeros(et_dmap.shape);
                topo_cp_weight_map_vis_b_gt = np.zeros(et_dmap.shape);
                topo_cp_weight_map_vis_d_gt = np.zeros(et_dmap.shape);
                topo_cp_weight_map_vis_b_perf = np.zeros(et_dmap.shape);
                topo_cp_weight_map_vis_d_perf = np.zeros(et_dmap.shape);
                topo_cp_weight_map_vis_grid = np.zeros(et_dmap.shape);
                topo_cp_ref_map = np.zeros(et_dmap.shape);

                gt_dmap_j = gt_dmap.detach().cpu().numpy();
                et_dmap_j = et_dmap.detach().cpu().numpy();
                min_x = max(10 , random.randint(0,int(topo_size/2)));
                min_y = max(10 , random.randint(0,int(topo_size/2)));
                max_x = -10 - topo_size;
                max_y = -10 - topo_size;
                et_sig = criterion_sig(et_dmap.squeeze(dim=1))
                for y in range(min_y, gt_dmap_j.shape[-2]+max_y, topo_size-2*sub_patch_border_width):
                    for x in range(min_x, gt_dmap_j.shape[-1]+max_x, topo_size-2*sub_patch_border_width):
                        #if(random.randint(0,1)==1):
                        #    continue 
                        topo_cp_weight_map_vis_grid[0,0,y,x] = 1
                        #print('y=',y)
                        #print('x=',x)
                        likelihood_sig = et_sig[:,y:min(y+topo_size, gt_dmap_j.shape[-2]), x:min(x+topo_size, gt_dmap_j.shape[-1])].squeeze();
                        likelihood = likelihood_sig.detach().cpu().numpy();
                        groundtruth = gt_dmap_j[:,:, y:min(y+topo_size, gt_dmap_j.shape[-2]), x:min(x+topo_size, gt_dmap_j.shape[-1])].squeeze();
                    
                        #print('likelihood.shape= ', likelihood.shape)
                        #print('groundtruth.shape=', groundtruth.shape)
                        if(len(likelihood.shape) < 2 or len(groundtruth.shape) < 2 ):
                            continue;
                        if(topo_size >= 100):
                            likelihood_2 = imresize(likelihood, (likelihood.shape[0]//2, likelihood.shape[1]//2)) 
                            if(likelihood_2.max() > 0):
                                likelihood_2 = likelihood_2/likelihood_2.max()*likelihood.max()
                            groundtruth_2 = imresize(groundtruth, (groundtruth.shape[0]//2, groundtruth.shape[1]//2))
                            if(groundtruth_2.max() > 0):
                                groundtruth_2 = groundtruth_2/groundtruth_2.max()*groundtruth.max()
                            pd_lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom_lh(-likelihood_2*mm, padwidth = padwidth, homo_dim=0)
                            pd_gt, bcp_gt, dcp_gt = compute_persistence_2DImg_1DHom_gt(-groundtruth_2*mm, padwidth = padwidth, homo_dim=0)
                            bcp_lh *= 2
                            dcp_lh *= 2
                            bcp_gt *= 2
                            dcp_gt *= 2
                        else:
                            pd_lh, bcp_lh, dcp_lh = compute_persistence_2DImg_1DHom_lh(-likelihood*mm, padwidth = padwidth, homo_dim=0)
                            pd_gt, bcp_gt, dcp_gt = compute_persistence_2DImg_1DHom_gt(-groundtruth*mm, padwidth = padwidth, homo_dim=0)
                        pers_thd_lh = 0.1
                        print('pd_lh.shape[0]',pd_lh.shape[0])
                        if(pd_lh.shape[0] > 0):
                            lh_pers = pd_lh[:, 1] - pd_lh[:, 0]
                            lh_pers_valid = lh_pers[np.where(lh_pers > pers_thd_lh)];
                        else:
                            lh_pers =np.array([])
                            lh_pers_valid = np.array([])

                        pers_thd_gt = 0.0
                        if(pd_gt.shape[0] > 0):
                            gt_pers = pd_gt[:, 1] - pd_gt[:, 0]
                            gt_pers_valid = gt_pers[np.where(gt_pers > pers_thd_gt)];
                        else:
                            gt_pers = np.array([])
                            gt_pers_valid = np.array([]);

                        using_lh_cp = True; 
                        if(pd_lh.shape[0] > gt_pers_valid.shape[0]): 
                            force_list, idx_holes_to_fix, idx_holes_to_remove, idx_holes_perfect = compute_dgm_force(pd_lh, pd_gt, pers_thresh=pers_thd_lh,pers_thresh_perfect=0.99, do_return_perfect=True)
                            n_fix += len(idx_holes_to_fix);
                            n_remove += len(idx_holes_to_remove);
                            print('len(idx_holes_to_fix)', len(idx_holes_to_fix))
                            print('len(idx_holes_to_remove)', len(idx_holes_to_remove))
                            print('len(idx_holes_perfect)', len(idx_holes_perfect))
                            if(len(idx_holes_to_fix)>0 or len(idx_holes_to_remove ) > 0):
                                for h in range(min(1000,len(idx_holes_perfect))):
                                    hole_indx = idx_holes_perfect[h];
                                    if(int(bcp_lh[hole_indx][0]) >= sub_patch_border_width and int(bcp_lh[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(bcp_lh[hole_indx][1]) >= sub_patch_border_width and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]-sub_patch_border_width):
                                        topo_cp_weight_map_vis_b_perf[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                                    if(int(dcp_lh[hole_indx][0]) >= sub_patch_border_width and int(dcp_lh[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(dcp_lh[hole_indx][1]) >= sub_patch_border_width and int(dcp_lh[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                                        topo_cp_weight_map_vis_d_perf[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood

                                for h in range(min(1000,len(idx_holes_to_fix))):
                                    hole_indx = idx_holes_to_fix[h];
                                    if(int(bcp_lh[hole_indx][0]) >= sub_patch_border_width and int(bcp_lh[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(bcp_lh[hole_indx][1]) >= sub_patch_border_width and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]-sub_patch_border_width):
                                        topo_cp_weight_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                                        topo_cp_weight_map_vis_b_fix[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                                        topo_cp_ref_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1;
                                    if(int(dcp_lh[hole_indx][0]) >= sub_patch_border_width and int(dcp_lh[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(dcp_lh[hole_indx][1]) >= sub_patch_border_width and int(dcp_lh[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                                        topo_cp_weight_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                                        topo_cp_weight_map_vis_d_fix[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                                        topo_cp_ref_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 0; 

                                for h in range(min(1000,len(idx_holes_to_remove))):
                                    hole_indx = idx_holes_to_remove[h];
                                    if(int(bcp_lh[hole_indx][0]) >= sub_patch_border_width and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] - sub_patch_border_width and int(bcp_lh[hole_indx][1]) >= sub_patch_border_width and int(bcp_lh[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                                        topo_cp_weight_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to death  # push to diagonal
                                        topo_cp_weight_map_vis_b_rem[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = 1; # push birth to death  # push to diagonal
                                        if(int(dcp_lh[hole_indx][0]) >= sub_patch_border_width and int(dcp_lh[hole_indx][0]) < likelihood.shape[0]- sub_patch_border_width and int(dcp_lh[hole_indx][1]) >= sub_patch_border_width and int(dcp_lh[hole_indx][1]) < likelihood.shape[1]- sub_patch_border_width):
                                            topo_cp_ref_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = likelihood[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]; 
                                        else:
                                            topo_cp_ref_map[0,0,y+int(bcp_lh[hole_indx][0]), x+int(bcp_lh[hole_indx][1])] = groundtruth[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])];  
                                    if(int(dcp_lh[hole_indx][0]) >= sub_patch_border_width and int(dcp_lh[hole_indx][0]) < likelihood.shape[0] - sub_patch_border_width and int(dcp_lh[hole_indx][1]) >= sub_patch_border_width and int(dcp_lh[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                                        topo_cp_weight_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to birth # push to diagonal
                                        topo_cp_weight_map_vis_d_rem[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = 1; # push death to birth # push to diagonal
                                        if(int(bcp_lh[hole_indx][0]) >= sub_patch_border_width and int(bcp_lh[hole_indx][0]) < likelihood.shape[0] - sub_patch_border_width and int(bcp_lh[hole_indx][1]) >= sub_patch_border_width and int(bcp_lh[hole_indx][1]) < likelihood.shape[1]- sub_patch_border_width):
                                            topo_cp_ref_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = likelihood[int(bcp_lh[hole_indx][0]), int(bcp_lh[hole_indx][1])]; 
                                        else:
                                            topo_cp_ref_map[0,0,y+int(dcp_lh[hole_indx][0]), x+int(dcp_lh[hole_indx][1])] = groundtruth[int(dcp_lh[hole_indx][0]), int(dcp_lh[hole_indx][1])]; 
                            if(len(idx_holes_to_fix) + len(idx_holes_perfect) < gt_pers_valid.shape[0]):
                                for hole_indx in range(gt_pers.shape[0]):
                                    if(int(bcp_gt[hole_indx][0]) >= sub_patch_border_width and int(bcp_gt[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(bcp_gt[hole_indx][1]) >= sub_patch_border_width and int(bcp_gt[hole_indx][1]) < likelihood.shape[1]-sub_patch_border_width):
                                        topo_cp_weight_map[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                                        topo_cp_weight_map_vis_b_gt[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                                        topo_cp_ref_map[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = groundtruth[int(bcp_gt[hole_indx][0]), int(bcp_gt[hole_indx][1])]; 
                                    if(int(dcp_gt[hole_indx][0]) >= sub_patch_border_width and int(dcp_gt[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(dcp_gt[hole_indx][1]) >= sub_patch_border_width and int(dcp_gt[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                                        topo_cp_weight_map[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                                        topo_cp_weight_map_vis_d_gt[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                                        topo_cp_ref_map[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = groundtruth[int(dcp_gt[hole_indx][0]), int(dcp_gt[hole_indx][1])]; 

                        else:
                            using_lh_cp = False;
                            for hole_indx in range(gt_pers.shape[0]):
                                if(int(bcp_gt[hole_indx][0]) >= sub_patch_border_width and int(bcp_gt[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(bcp_gt[hole_indx][1]) >= sub_patch_border_width and int(bcp_gt[hole_indx][1]) < likelihood.shape[1]-sub_patch_border_width):
                                    topo_cp_weight_map[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                                    topo_cp_weight_map_vis_b_gt[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = 1; # push birth to 0 i.e. min birth prob or likelihood
                                    topo_cp_ref_map[0,0,y+int(bcp_gt[hole_indx][0]), x+int(bcp_gt[hole_indx][1])] = groundtruth[int(bcp_gt[hole_indx][0]), int(bcp_gt[hole_indx][1])]; 
                                if(int(dcp_gt[hole_indx][0]) >= sub_patch_border_width and int(dcp_gt[hole_indx][0]) < likelihood.shape[0]-sub_patch_border_width and int(dcp_gt[hole_indx][1]) >= sub_patch_border_width and int(dcp_gt[hole_indx][1]) < likelihood.shape[1] - sub_patch_border_width):
                                    topo_cp_weight_map[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                                    topo_cp_weight_map_vis_d_gt[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = 1; # push death to 1 i.e. max death prob or likelihood
                                    topo_cp_ref_map[0,0,y+int(dcp_gt[hole_indx][0]), x+int(dcp_gt[hole_indx][1])] = groundtruth[int(dcp_gt[hole_indx][0]), int(dcp_gt[hole_indx][1])]; 
                topo_cp_weight_map = torch.tensor(topo_cp_weight_map, dtype=torch.float).to(device)
                topo_cp_ref_map = torch.tensor(topo_cp_ref_map, dtype=torch.float).to(device)

                print('topo_cp_ref_map.sum()',topo_cp_ref_map.sum())
                intersection = (et_sig * topo_cp_ref_map*topo_cp_weight_map).sum()
                union = ((et_sig*topo_cp_weight_map.squeeze(dim=1))**2).sum() + ((topo_cp_ref_map)**2).sum()
                loss_pers =  1 - ((2 * intersection + 1) / (union + 1))

                if(i%50==0): 
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_img'+'.png'), (img.squeeze().detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_gt'+'.png'), (gt_dmap.squeeze().detach().cpu().numpy()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_likelihood'+'.png'), (criterion_sig(et_dmap).squeeze().detach().cpu().numpy()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp'+'.png'), (topo_cp_weight_map.squeeze().detach().cpu().numpy()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp_b_fix'+'.png'), (topo_cp_weight_map_vis_b_fix.squeeze()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp_d_fix'+'.png'), (topo_cp_weight_map_vis_d_fix.squeeze()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp_b_rem'+'.png'), (topo_cp_weight_map_vis_b_rem.squeeze()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp_d_rem'+'.png'), (topo_cp_weight_map_vis_d_rem.squeeze()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp_b_gt'+'.png'), (topo_cp_weight_map_vis_b_gt.squeeze()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp_d_gt'+'.png'), (topo_cp_weight_map_vis_d_gt.squeeze()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp_b_perf'+'.png'), (topo_cp_weight_map_vis_b_perf.squeeze()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp_d_perf'+'.png'), (topo_cp_weight_map_vis_d_perf.squeeze()*255).astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_cp_grid'+'.png'), (topo_cp_weight_map_vis_grid.squeeze()*255).astype(np.uint8));

            if(not(lamda_pers > 0 and epoch >= epoch_start_pers_loss) and i%50==0): 
                io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_img'+'.png'), (img.squeeze().detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8));
                io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_gt'+'.png'), (gt_dmap.squeeze().detach().cpu().numpy()*255).astype(np.uint8));
                io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_train'+ '_indx'+str(i)+'_likelihood'+'.png'), (criterion_sig(et_dmap).squeeze().detach().cpu().numpy()*255).astype(np.uint8));

            et_sig = criterion_sig(et_dmap.squeeze(dim=1))
            intersection = (et_sig * gt_dmap ).sum()
            union = (et_sig**2).sum() + (gt_dmap**2).sum()
            loss_dice =  1 - ((2 * intersection + 1) / (union + 1))


            loss = lamda_dice * loss_dice + lamda_pers * loss_pers 

            e_hard = filters.apply_hysteresis_threshold(et_sig.detach().cpu().numpy().squeeze(), thresh_low, thresh_high)
            e_hard2 = (e_hard > 0).astype(np.uint8)
            comp_mask = label(e_hard2)
            e_count = comp_mask.max()
            gt_dots = gt_dots.detach().cpu().numpy().squeeze()
            g_count = gt_dots.sum()
            err= e_count - g_count
            mae += abs(err)
            rmse += err**2

            mae += abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            rmse += ((et_dmap.data.sum()-gt_dmap.data.sum())**2).item()

            epoch_loss_dice += loss_dice.item()
            epoch_loss_pers += loss_pers.item()
    
            print("epoch:",epoch, " train batch i:",i, 'loss_dice=',loss_dice.item(), 'loss_pers=',loss_pers.item())
            log_file.write("epoch: {}  train batch i: {} loss_dice= {} loss_pers= {} count_err {}\n".format(epoch, i, loss_dice.item(), loss_pers.item(), err))
            log_file.flush()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sys.stdout.flush();
    
        epoch_loss_dice /= len(train_loader)
        epoch_loss_pers /= len(train_loader)
        mae /= len(train_loader)
        rmse /= len(train_loader)


        sys.stdout.flush();      
  

        train_loss_dice_list.append(epoch_loss_dice)
        train_loss_pers_list.append(epoch_loss_pers)
        train_mae_list.append(mae)
        train_rmse_list.append(rmse)
        train_rmse_mae_list.append(rmse*2+mae)

        np.array(train_loss_dice_list).astype(np.float16).dump(os.path.join(checkpoints_save_path, 'epochs_train_loss_dice_list.npy'))
        np.array(train_loss_pers_list).astype(np.float16).dump(os.path.join(checkpoints_save_path, 'epochs_train_loss_pers_list.npy'))
        np.array(train_mae_list).astype(np.float16).dump(os.path.join(checkpoints_save_path, 'epochs_train_mae_list.npy'))
        np.array(train_rmse_list).astype(np.float16).dump(os.path.join(checkpoints_save_path, 'epochs_train_rmse_list.npy'))
        np.array(train_rmse_mae_list).astype(np.float16).dump(os.path.join(checkpoints_save_path, 'epochs_train_2rmse_plus_mae_list.npy'))

        print("epoch:",epoch, " train epoch_loss_dice:",epoch_loss_dice, 'epoch_loss_pers', epoch_loss_pers, 'mae', mae, 'rmse', rmse, '2rmse+mae', rmse*2+mae)
        log_file.write("epoch: {}  train epoch_loss_dice {} epoch_loss_pers {} mae {} rmse {} rmse*2+mae {} \n".format(epoch, epoch_loss_dice, epoch_loss_pers, mae, rmse, rmse*2+mae))
        log_file.flush()

        #torch.save(model.state_dict(),'./checkpoints/epoch_'+str(epoch)+".pth")
    
        # testing phase
        model.eval()
        epoch_loss_dice=0
        mae=0;
        rmse=0
        loss_val = 0
        for i,(img,gt_dmap, gt_dots) in enumerate(tqdm(test_loader)):
            if(test_patch_size > 0 and i > 5): # because test_patch_size > 0 need to run a separate test to evaluate models on val/test data to find optimized model, so do not need to run on all val/test data, a sample to visualize is enough.
                break;
            img=img.to(device)
            gt_dmap = gt_dmap > 0
            gt_dmap = gt_dmap.type(torch.FloatTensor)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img)[:,:,2:-2,2:-2]
            et_sig = criterion_sig(et_dmap.squeeze(dim=1))
            intersection = (et_sig * gt_dmap ).sum()
            union = (et_sig**2).sum() + (gt_dmap**2).sum()
            loss_dice =  1 - ((2 * intersection + 1) / (union + 1))
            #print('loss_dice=',loss_dice.item())
            loss_val += loss_dice.item()
            epoch_loss_dice += loss_dice.item()


            if(i <6):
                io.imsave(os.path.join(checkpoints_save_path, 'test'+ '_indx'+str(i)+'_img'+'.png'), (img.squeeze().detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8));
                io.imsave(os.path.join(checkpoints_save_path, 'test'+ '_indx'+str(i)+'_gt'+'.png'), (gt_dmap.squeeze().detach().cpu().numpy()*255).astype(np.uint8));
                io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_test'+ '_indx'+str(i)+'_likelihood'+'_loss_'+ "{:.4f}".format(loss_dice.item()) + '_err'+str(err)+'.png'), (criterion_sig(et_dmap).squeeze().detach().cpu().numpy()*255).astype(np.uint8));

            e_hard = filters.apply_hysteresis_threshold(et_sig.detach().cpu().numpy().squeeze(), thresh_low, thresh_high)
            e_hard2 = (e_hard > 0).astype(np.uint8)
            comp_mask = label(e_hard2)
            e_count = comp_mask.max()
            gt_dots = gt_dots.detach().cpu().numpy().squeeze()
            g_count = gt_dots.sum()
            err= e_count - g_count
            mae += abs(err)
            rmse += err**2

            print("epoch:",epoch, " test batch i:",i, 'loss_dice=',loss_dice.item())
            log_file.write("epoch: {}  train batch i: {} loss_dice= {} count_err {}\n".format(epoch, i, loss_dice.item(), err))
            log_file.flush()

            del img,gt_dmap, et_dmap, et_sig, gt_dots
        saved = False

        epoch_loss_dice /= len(test_loader)
        mae /= len(test_loader)
        rmse /= len(test_loader)
        rmse_mae = 2*rmse+mae

        test_loss_dice_list.append(epoch_loss_dice)
        test_mae_list.append(mae)
        test_rmse_list.append(rmse)
        test_rmse_mae_list.append(rmse_mae)

        np.array(test_loss_dice_list).astype(np.float16).dump(os.path.join(checkpoints_save_path, 'epochs_test_loss_dice_list.npy'))
        np.array(test_mae_list).astype(np.float16).dump(os.path.join(checkpoints_save_path, 'epochs_test_mae_list.npy'))
        np.array(test_rmse_list).astype(np.float16).dump(os.path.join(checkpoints_save_path, 'epochs_test_rmse_list.npy'))
        np.array(test_rmse_mae_list).astype(np.float16).dump(os.path.join(checkpoints_save_path, 'epochs_test_2rmse_plus_mae_list.npy'))

        
        if rmse_mae<=min_rmse_mae:
            min_rmse_mae=rmse_mae
            min_epoch_rmse_mae =epoch
            print('best test rmse_mae epoch',epoch, rmse_mae)
            log_file.write("best test rmse_mae {} epoch {} \n".format(rmse_mae, epoch))
            log_file.flush()
            if(not saved):
                #torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+"_best_2rmse_plus_mae.pth")) # save only if get better error                
                torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+".pth")) # save only if get better error                
                saved = True
        if rmse<=min_rmse:
            min_rmse=rmse
            min_epoch_rmse =epoch
            print('best test rmse epoch',epoch, rmse)
            log_file.write("best test rmse {} epoch {} \n".format(rmse, epoch))
            log_file.flush()
            if(not saved):
                #torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+"_best_rmse.pth")) # save only if get better error                
                torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+".pth")) # save only if get better error                
                saved = True
        if mae<=min_mae:
            min_mae=mae
            min_epoch_mae=epoch
            print('best test mae epoch',epoch, mae)
            log_file.write("best test mae {} epoch {} \n".format(mae, epoch))
            log_file.flush()
            if(not saved):
                #torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+"_best_mae.pth")) # save only if get better error                
                torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+".pth")) # save only if get better error                
                saved = True
        if epoch_loss_dice<=min_loss:
            min_loss=epoch_loss_dice
            min_epoch_loss =epoch
            print('best test dice loss epoch',epoch, epoch_loss_dice)
            log_file.write("best test dice loss {} epoch {} \n".format(epoch_loss_dice, epoch))
            log_file.flush()
            if(not saved):
                #torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+"_best_dice.pth")) # save only if get better error                
                torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+".pth")) # save only if get better error                
                saved = True
        if(not saved and test_patch_size > 0): # because test_patch_size > 0 need to run a separate test to evaluate models on val/test data to find optimized model
            torch.save(model.state_dict(),os.path.join(checkpoints_save_path, 'epoch_'+str(epoch)+"_tmp"+".pth")) # save only if get better error
            saved = True

        print("epoch:",epoch, " test epoch_loss_dice:",epoch_loss_dice, 'mae', mae, 'rmse', rmse, '2rmse_plus_mae', rmse_mae)
        log_file.write("epoch {} test epoch_loss_dice {} mae {} rmse {} 2rmse_plus_mae {} \n".format(epoch, epoch_loss_dice, mae, rmse, rmse_mae))
        log_file.flush()
        sys.stdout.flush();


    sys.stdout.flush();
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    