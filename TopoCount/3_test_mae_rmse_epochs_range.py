#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import matplotlib.cm as CM
import os
import numpy as np
from skimage import io;
import glob;
import cv2 ; 
import sys;
#from scipy.misc import imresize
#from scipy.ndimage.filters import convolve    
from skimage.measure import label
#import pickle;
from skimage import filters
import math
from tqdm import tqdm as tqdm
import time
import torch
import torch.nn as nn
#from scipy import ndimage

from unet_vgg4_cc import UnetVggCC
from my_dataset_test import CrowdDataset


'''
Generate calculate mae and rmse for a range of saved epochs
Output files:
out_.txt: stats per image and over all images.

In main set epochs range: 
    start_epoch
    end_epoch

In main are some default configurations for the datasets: ShanghaTech Part A, ShanghaTech Part B, UCF-QNRF, JHU++, NWPU-Crowd.
Uncomment the approporiate configuration
'''

def run_eval_model(model_param_path, test_loader, out_dir, log_file, thresh_low=0.4, thresh_high=0.5, size_thresh=-1, model_name='topocount_best'):
    global model
    global device
    criterion_sig = nn.Sigmoid() # initialize sigmoid layer
    model.load_state_dict(torch.load(model_param_path), strict=True);
    model.to(device)

    mae = 0
    rmse = 0
    pos = 0
    neg = 0
    files_count = 0
    #log_file.write("modelname {} \n".format(model_name))
    #log_file.flush()
    for i,(img,gt_dots,img_name) in enumerate(tqdm(test_loader, disable=True)):
        sys.stdout.flush()
        files_count += 1
        img_name = img_name[0]
        print('img', img_name, img.shape)
        sys.stdout.flush();
        if(img.shape[-1]<2048 and img.shape[-2]<2048):
            img=img.to(device)
            # forward propagation
            et_dmap=model(img)[:,:,2:-2,2:-2]
            et_sig = criterion_sig(et_dmap.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap=et_dmap.squeeze().detach().cpu().numpy()
        elif(img.shape[-2] * img.shape[-1] > 30000): # divide image into 3x3 overlapping to fit in GPU
            et_dmap = 0
            et_sig = np.zeros((img.shape[-2], img.shape[-1]))
            et_dmap = np.zeros((img.shape[-2], img.shape[-1]))
            y_part = img.shape[-2]//3
            x_part = img.shape[-1]//3
            overlap_y = 96
            overlap_x = 96
            max_scale = 16
            ds_rows=int(y_part//max_scale)*max_scale
            ds_cols=int(x_part//max_scale)*max_scale
            overlap_y += (max_scale - (y_part - ds_rows))
            overlap_x += (max_scale - (x_part - ds_cols))
            #print('#quad 0,0')
            #quad 0,0
            img_sub = img[:,:,:y_part+overlap_y, :x_part+overlap_x].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            et_sig[:y_part, :x_part] = et_sig_sub[:y_part, :x_part]
            et_dmap[:y_part, :x_part] = et_dmap_sub[:y_part, :x_part]
            del img_sub, et_dmap_sub, et_sig_sub
            #quad 0,1
            img_sub = img[:,:,:y_part+overlap_y, x_part-overlap_x//2:2*x_part+overlap_x//2 ].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            et_sig[:y_part, x_part:2*x_part] = et_sig_sub[:y_part, overlap_x//2:overlap_x//2 + x_part]
            et_dmap[:y_part, x_part:2*x_part] = et_dmap_sub[:y_part, overlap_x//2:overlap_x//2 + x_part]
            del img_sub, et_dmap_sub, et_sig_sub
            #quad 0,2
            img_sub = img[:,:,:y_part+overlap_y, 2*x_part-overlap_x: ].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            et_sig[:y_part, -x_part:] = et_sig_sub[:y_part, -x_part:]
            et_dmap[:y_part, -x_part:] = et_dmap_sub[:y_part, -x_part:]
            del img_sub, et_dmap_sub, et_sig_sub
            #quad 1,0
            img_sub = img[:,:,y_part-overlap_y//2:2*y_part+overlap_y//2, :x_part+overlap_x].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            et_sig[y_part:2*y_part, :x_part] = et_sig_sub[overlap_y//2:overlap_y//2 + y_part, :x_part]
            et_dmap[y_part:2*y_part, :x_part] = et_dmap_sub[overlap_y//2:overlap_y//2 + y_part, :x_part]
            del img_sub, et_dmap_sub, et_sig_sub
            #quad 1,1
            img_sub = img[:,:,y_part-overlap_y//2:2*y_part+overlap_y//2, x_part-overlap_x//2:2*x_part+overlap_x//2 ].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            et_sig[y_part:2*y_part, x_part:2*x_part] = et_sig_sub[overlap_y//2:overlap_y//2 + y_part, overlap_x//2:overlap_x//2 + x_part]
            et_dmap[y_part:2*y_part, x_part:2*x_part] = et_dmap_sub[overlap_y//2:overlap_y//2 + y_part, overlap_x//2:overlap_x//2 + x_part]
            del img_sub, et_dmap_sub, et_sig_sub
            #quad 1,2
            img_sub = img[:,:,y_part-overlap_y//2:2*y_part+overlap_y//2, 2*x_part-overlap_x: ].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            et_sig[y_part:2*y_part, -x_part:] = et_sig_sub[overlap_y//2:overlap_y//2 + y_part, -x_part:]
            et_dmap[y_part:2*y_part, -x_part:] = et_dmap_sub[overlap_y//2:overlap_y//2 + y_part, -x_part:]
            del img_sub, et_dmap_sub, et_sig_sub
            #quad 2,0
            img_sub = img[:,:,2*y_part-overlap_y:, :x_part+overlap_x].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            et_sig[-y_part:, :x_part] = et_sig_sub[-y_part:, :x_part]
            et_dmap[-y_part:, :x_part] = et_dmap_sub[-y_part:, :x_part]
            del img_sub, et_dmap_sub, et_sig_sub
            #quad 2,1
            img_sub = img[:,:,2*y_part-overlap_y:, x_part-overlap_x//2:2*x_part+overlap_x//2 ].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            et_sig[-y_part:, x_part:2*x_part] = et_sig_sub[-y_part:, overlap_x//2:overlap_x//2 + x_part]
            et_dmap[-y_part:, x_part:2*x_part] = et_dmap_sub[-y_part:, overlap_x//2:overlap_x//2 + x_part]
            del img_sub, et_dmap_sub, et_sig_sub
            #quad 2,2
            img_sub = img[:,:,2*y_part-overlap_y:, 2*x_part-overlap_x: ].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            et_sig[-y_part:, -x_part:] = et_sig_sub[-y_part:, -x_part:]
            et_dmap[-y_part:, -x_part:] = et_dmap_sub[-y_part:, -x_part:]
            del img_sub, et_dmap_sub, et_sig_sub
        else: # divide image into 2x2 overlapping to fit in GPU
            et_dmap = 0
            et_sig = np.zeros((img.shape[-2], img.shape[-1]))
            et_dmap = np.zeros((img.shape[-2], img.shape[-1]))
            y_half = img.shape[-2]//2
            x_half = img.shape[-1]//2
            overlap_y = 96
            overlap_x = 96
            max_scale = 16
            ds_rows=int(img.shape[-2]//max_scale)*max_scale
            ds_cols=int(img.shape[-1]//max_scale)*max_scale
            overlap_y += (max_scale - (img.shape[-2] - ds_rows))
            overlap_x += (max_scale - (img.shape[-1] - ds_cols))
            #print('#quad 0,0')
            #print('mem',torch.cuda.memory_allocated(device))
            #quad 0,0
            img_sub = img[:,:,:y_half+overlap_y, :x_half+overlap_x].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            img_sub = img_sub.squeeze().detach().cpu().numpy()
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print(':x_half',x_half)
            if(et_sig_sub.shape[0] > y_half + overlap_y):
                et_sig_sub = et_sig_sub[ (et_sig_sub.shape[0]-y_half - overlap_y)//2:-(et_sig_sub.shape[0]-y_half - overlap_y)//2 ,:]
                et_dmap_sub = et_dmap_sub[ (et_dmap_sub.shape[0]-y_half - overlap_y)//2:-(et_dmap_sub.shape[0]-y_half - overlap_y)//2 ,:]
            if(et_sig_sub.shape[1] > x_half + overlap_x):
                et_sig_sub = et_sig_sub[ :, (et_sig_sub.shape[1]-x_half - overlap_x)//2:-(et_sig_sub.shape[1]-x_half - overlap_x)//2]
                et_dmap_sub = et_dmap_sub[ :, (et_dmap_sub.shape[1]-x_half - overlap_x)//2:-(et_dmap_sub.shape[1]-x_half - overlap_x)//2]
            et_sig[:y_half, :x_half] = et_sig_sub[:y_half, :x_half]
            et_dmap[:y_half, :x_half] = et_dmap_sub[:y_half, :x_half]
            del img_sub, et_dmap_sub, et_sig_sub
            torch.cuda.empty_cache()
            #print('#quad 0,1')
            #print('mem',torch.cuda.memory_allocated(device))
            #quad 0,1
            img_sub = img[:,:,:y_half+overlap_y, x_half-overlap_x:].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print(':y_half',y_half)
            #print('x_half:',x_half, overlap_x)
            if(et_sig_sub.shape[0] > y_half + overlap_y):
                et_sig_sub = et_sig_sub[ (et_sig_sub.shape[0]-y_half - overlap_y)//2:-(et_sig_sub.shape[0]-y_half - overlap_y)//2 ,:]
                et_dmap_sub = et_dmap_sub[ (et_dmap_sub.shape[0]-y_half - overlap_y)//2:-(et_dmap_sub.shape[0]-y_half - overlap_y)//2 ,:]
            if(et_sig_sub.shape[1] > x_half + overlap_x):
                et_sig_sub = et_sig_sub[ :, (et_sig_sub.shape[1]-x_half - overlap_x)//2:-(et_sig_sub.shape[1]-x_half - overlap_x)//2]
                et_dmap_sub = et_dmap_sub[ :, (et_dmap_sub.shape[1]-x_half - overlap_x)//2:-(et_dmap_sub.shape[1]-x_half - overlap_x)//2]
            et_sig[:y_half, x_half:] = et_sig_sub[:y_half, overlap_y:]
            et_dmap[:y_half, x_half:] = et_dmap_sub[:y_half, overlap_y:]
            del img_sub, et_dmap_sub, et_sig_sub
            torch.cuda.empty_cache()
            #print('#quad 1,0')
            #print('mem',torch.cuda.memory_allocated(device))
            #quad 1,0
            img_sub = img[:,:,y_half-overlap_y:, :x_half+overlap_x].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print('y_half:',y_half, overlap_y)
            #print(':x_half',x_half)
            if(et_sig_sub.shape[0] > y_half + overlap_y):
                et_sig_sub = et_sig_sub[ (et_sig_sub.shape[0]-y_half - overlap_y)//2:-(et_sig_sub.shape[0]-y_half - overlap_y)//2 ,:]
                et_dmap_sub = et_dmap_sub[ (et_dmap_sub.shape[0]-y_half - overlap_y)//2:-(et_dmap_sub.shape[0]-y_half - overlap_y)//2 ,:]
            if(et_sig_sub.shape[1] > x_half + overlap_x):
                et_sig_sub = et_sig_sub[ :, (et_sig_sub.shape[1]-x_half - overlap_x)//2:-(et_sig_sub.shape[1]-x_half - overlap_x)//2]
                et_dmap_sub = et_dmap_sub[ :, (et_dmap_sub.shape[1]-x_half - overlap_x)//2:-(et_dmap_sub.shape[1]-x_half - overlap_x)//2]
            et_sig[y_half:, :x_half] = et_sig_sub[overlap_y:, :x_half]
            et_dmap[y_half:, :x_half] = et_dmap_sub[overlap_y:, :x_half]
            del img_sub, et_dmap_sub, et_sig_sub
            torch.cuda.empty_cache()
            #print('#quad 1,1')
            #print('mem',torch.cuda.memory_allocated(device))
            #quad 1,1
            img_sub = img[:,:,y_half-overlap_y:, x_half-overlap_x:].to(device)
            #print('img_sub',img_sub.shape)
            #et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
            et_dmap_sub=model(img_sub)
            #print('et_dmap_sub',et_dmap_sub.shape)
            et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
            et_dmap_sub=et_dmap_sub.squeeze().detach().cpu().numpy()
            #print('et_sig_sub',et_sig_sub.shape)
            #print('y_half:',y_half, overlap_y)
            #print('x_half:',x_half, overlap_x)
            if(et_sig_sub.shape[0] > y_half + overlap_y):
                et_sig_sub = et_sig_sub[ (et_sig_sub.shape[0]-y_half - overlap_y)//2:-(et_sig_sub.shape[0]-y_half - overlap_y)//2 ,:]
                et_dmap_sub = et_dmap_sub[ (et_dmap_sub.shape[0]-y_half - overlap_y)//2:-(et_dmap_sub.shape[0]-y_half - overlap_y)//2 ,:]
            if(et_sig_sub.shape[1] > x_half + overlap_x):
                et_sig_sub = et_sig_sub[ :, (et_sig_sub.shape[1]-x_half - overlap_x)//2:-(et_sig_sub.shape[1]-x_half - overlap_x)//2]
                et_dmap_sub = et_dmap_sub[ :, (et_dmap_sub.shape[1]-x_half - overlap_x)//2:-(et_dmap_sub.shape[1]-x_half - overlap_x)//2]
            et_sig[y_half:, x_half:] = et_sig_sub[overlap_y:, overlap_x:]
            et_dmap[y_half:, x_half:] = et_dmap_sub[overlap_y:, overlap_x:]
            del img_sub, et_dmap_sub, et_sig_sub
            torch.cuda.empty_cache()

        #print('mem',torch.cuda.memory_allocated(device))
        gt_dots = gt_dots.detach().cpu().numpy().squeeze()
        g_count = gt_dots.sum()
        img = img.detach().cpu().numpy()
        #print('mem',torch.cuda.memory_allocated(device))


        e_hard = filters.apply_hysteresis_threshold(et_sig, thresh_low, thresh_high)
        e_hard2 = (e_hard > 0).astype(np.uint8)
        comp_mask = label(e_hard2)
        e_count = comp_mask.max()
        s_count=0
        if(size_thresh > 0):
            for c in range(1,comp_mask.max()+1):
                s = (comp_mask == c).sum()
                if(s < size_thresh):
                    e_count -=1
                    s_count +=1

        # get dot predictions from topological map (centers of connected components)
        e_dot = np.zeros(gt_dots.shape)
        e_dot_vis = np.zeros(gt_dots.shape)
        contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for idx in range(len(contours)):
            contour_i = contours[idx]
            M = cv2.moments(contour_i)
            if(M['m00'] == 0):
                continue;
            cx = round(M['m10'] / M['m00'])
            cy = round(M['m01'] / M['m00'])
            e_dot_vis[cy-1:cy+1, cx-1:cx+1] = 1
            e_dot[cy, cx] = 1
        io.imsave(os.path.join(out_dir, img_name + '_e_centers_'+'.png'), (e_dot_vis*255).astype(np.uint8))
        e_dot.astype(np.uint8).dump(os.path.join(out_dir, img_name + '_e_centers_'+'.npy'))


        err= e_count - g_count
        mae += abs(err)
        rmse += err**2
        if(err > 0):
            pos += err
        else:
            neg -= err
        #print(img_name, e_count, g_count, err)
        log_file.write("image {} e_count {} g_count {} err {} \n".format(img_name, e_count, g_count, err))
        log_file.flush()
        sys.stdout.flush();
        #io.imsave(os.path.join(out_dir, img_name), (img.squeeze().detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8));
        io.imsave(os.path.join(out_dir, img_name.replace('.jpg','_likelihood'+'.png')), (et_sig*255).astype(np.uint8));
        io.imsave(os.path.join(out_dir, img_name.replace('.jpg','_hard_'+str(thresh_low)+'-'+str(thresh_high)+'_err'+str(err)+'.png')), (e_hard2*255).astype(np.uint8));

        et_dmap.astype(np.float16).dump(os.path.join(out_dir, img_name.replace('.jpg','_raw'+'.npy')))
        et_sig.astype(np.float16).dump(os.path.join(out_dir, img_name.replace('.jpg','_likelihood'+'.npy')))
        e_hard2.astype(np.uint8).dump(os.path.join(out_dir, img_name.replace('.jpg','_hard_'+str(thresh_low)+'-'+str(thresh_high)+'_err'+str(err)+'.npy')))

        del img,gt_dots, et_dmap, et_sig
        torch.cuda.empty_cache()

    mae /= files_count
    rmse = math.sqrt(rmse/files_count)

    print('modelname', model_name, 'mae', mae, 'rmse', rmse, '2rmse+mae',2*rmse+mae, 'pos_err', pos, 'neg_err', neg)
    log_file.write("modelname {} mae {} rmse {} 2rmse+mae {} pos_err {} neg_err {} \n".format(model_name, mae, rmse, 2*rmse+mae, pos, neg))
    log_file.flush()
    sys.stdout.flush();

if __name__=="__main__":

    # Below are some default configurations for the datasets: ShanghaTech Part A, ShanghaTech Part B, UCF-QNRF, JHU++, NWPU-Crowd.
    # Uncomment the approporiate configuration
    ####################################################################################
    ## Configuration for ShanghaiTech Part A - Test
    ####################################################################################
    '''
    models_root_dir = './checkpoints/sh_parta_custom_topo1_patch50'
    #model_filename = 'topocount_sh_parta.pth'
    #out_dir = './eval/parta_custom_topo1_patch50_e48_test'
    out_dir = './eval/sh_parta_custom_topo1_patch50_topocount_test_epochs'
    out_filename = 'out.txt'
    root = './datasets/ShanghaiTech/'
    test_image_root = os.path.join(root,'part_A/test_data','images') 
    test_dots_root = os.path.join(root,'part_A/test_data','ground-truth_dots')   
    test_split_txt_filepath = None
    start_epoch=0
    end_epoch=5
    '''
    #####################################################################################
    ### Configuration for ShanghaiTech Part B - Test
    #####################################################################################
    #'''
    models_root_dir = './checkpoints/sh_partb_custom_topo1_patch50'
    #model_filename = 'topocount_sh_partb.pth'
    #out_dir = './eval/parta_custom_topo1_patch50_e48_test'
    out_dir = './eval/sh_partb_custom_topo1_patch50_topocount_test_epochs'
    out_filename = 'out.txt'
    root = './datasets/ShanghaiTech/'
    test_image_root = os.path.join(root,'part_B/test_data','images') 
    test_dots_root = os.path.join(root,'part_B/test_data','ground-truth_dots')   
    test_split_txt_filepath = None
    start_epoch=0
    end_epoch=5
    #'''
    #####################################################################################
    ### Configuration for UCF-QNRF - Test
    #####################################################################################
    '''
    models_root_dir = './checkpoints/qnrf_custom_topo1_patch100'
    #model_filename = 'topocount_qnrf.pth'
    #out_dir = './eval/qnrf_custom_topo1_patch100_e48_test'
    out_dir = './eval/qnrf_custom_topo1_patch100_topocount_test_epochs'
    out_filename = 'out.txt'
    root = './datasets/UCF-QNRF_ECCV18/UCF-QNRF_ECCV18'
    test_image_root = os.path.join(root,'Test','img_scaleshort2048')
    test_dots_root = os.path.join(root,'Test','ground-truth_dots_scaleshort2048')   
    test_split_txt_filepath = None
    start_epoch=0
    end_epoch=5
    '''
    #####################################################################################
    ### Configuration for JHU++ - Validation
    #####################################################################################
    '''
    models_root_dir = './checkpoints/jhu_custom_topo1_patch100'
    #model_filename = 'topocount_jhu.pth'
    #out_dir = './eval/jhu_custom_topo1_patch100_e48_val'
    out_dir = './eval/jhu_custom_topo1_patch100_topocount_val_epochs'
    out_filename = 'out.txt'
    root = './datasets/jhu/jhu_crowd_v2.0'
    test_image_root = os.path.join(root, 'val','images')
    test_dots_root = os.path.join(root, 'val','ground-truth_dots')    
    test_split_txt_filepath = None
    start_epoch=0
    end_epoch=5
    '''

    #####################################################################################
    ### Configuration for JHU++ - Test
    #####################################################################################
    '''
    models_root_dir = './checkpoints/jhu_custom_topo1_patch100'
    #model_filename = 'topocount_jhu.pth'
    #out_dir = './eval/jhu_custom_topo1_patch100_e48_test'
    out_dir = './eval/jhu_custom_topo1_patch100_topocount_test_epochs'
    out_filename = 'out.txt'
    root = './datasets/jhu/jhu_crowd_v2.0'
    test_image_root = os.path.join(root, 'test','images')
    test_dots_root = os.path.join(root, 'test','ground-truth_dots')    
    test_split_txt_filepath = None
    start_epoch=0
    end_epoch=5
    '''

    #####################################################################################
    ### Configuration for NWPU-Crowd - Validation
    #####################################################################################
    '''
    models_root_dir = './checkpoints/nwpu_custom_topo1_patch100'
    #model_filename = 'topocount_nwpu.pth'
    #out_dir = './eval/nwpu_custom_topo1_patch100_e48_val'
    out_dir = './eval/nwpu_custom_topo1_patch100_topocount_val_epochs'
    out_filename = 'out.txt'
    root = './datasets/nwpu-crowd'
    test_image_root = os.path.join(root,'images')
    test_dots_root = os.path.join(root,'ground-truth_dots') 
    test_split_txt_filepath = os.path.join(root,'val.txt')    
    start_epoch=0
    end_epoch=5
    '''    

    #####################################################################################
    ### Configuration for NWPU-Crowd - Test
    #####################################################################################
    '''
    models_root_dir = './checkpoints/nwpu_custom_topo1_patch100'
    #model_filename = 'topocount_nwpu.pth'
    #out_dir = './eval/nwpu_custom_topo1_patch100_e48_test'
    out_dir = './eval/nwpu_custom_topo1_patch100_topocount_test_epochs'
    out_filename = 'out.txt'
    root = './datasets/nwpu-crowd'
    test_image_root = os.path.join(root,'images')
    test_dots_root = os.path.join(root,'ground-truth_dots') 
    test_split_txt_filepath = os.path.join(root,'test.txt')    
    start_epoch=0
    end_epoch=5
    '''

    ####################################################################################

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    #thresh_low = 0.4
    #thresh_high = 0.5
    #size_thresh = -1

    gpu_or_cpu='cuda' # use cuda or cpu
    dropout_keep_prob = 1.0
    initial_pad = 126
    interpolate = 'False'
    conv_init = 'he'
    n_classes = 1
    n_channels = 1

    #if not os.path.exists(out_dir):
    #    os.mkdir(out_dir)

    device=torch.device(gpu_or_cpu)
    model=UnetVggCC(kwargs={'dropout_keep_prob':dropout_keep_prob, 'initial_pad':initial_pad, 'interpolate':interpolate, 'conv_init':conv_init, 'n_classes':n_classes, 'n_channels':n_channels})
    model.to(device)

    test_dataset=CrowdDataset(test_image_root, test_dots_root, split_txt_filepath=test_split_txt_filepath, phase='test', normalize=False)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

    with open(os.path.join(out_dir, out_filename), 'a+') as log_file:
        for epoch in range(start_epoch, end_epoch+1):
            print('test epoch ' + str(epoch) )
            model_param_path = os.path.join(models_root_dir, 'epoch_'+str(epoch)+'.pth');
            if(not os.path.isfile(model_param_path )): 
                print('not found ', model_param_path )
                model_param_path = os.path.join(models_root_dir, 'epoch_'+str(epoch)+'_tmp.pth');
                if(not os.path.isfile(model_param_path )): 
                    print('not found ', model_param_path )
                    continue;

            run_eval_model(model_param_path, test_loader, out_dir, log_file)
            log_file.flush()

    print('Done.')
    print('Check output in: ', os.path.join(out_dir, out_filename))
