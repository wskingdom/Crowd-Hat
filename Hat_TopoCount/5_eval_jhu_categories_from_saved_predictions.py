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
from skimage import filters
import math
from tqdm import tqdm as tqdm
import torch
import torch.nn as nn
#from scipy import ndimage

#from unet_vgg4_cc import UnetVggCC
#from my_dataset_highres4_jhu_wdots_wname import CrowdDataset
#from TDFMain_pytorch import *

'''
Evaluate the error for each category in the JHU++ dataset: low, medium, high, and weather

Output files:
out_jhu_categories.txt: mae and rmse per category.

In main modify the directories:  
data_dir: path contains prediction files.
root: dataset root.
gt_dir: ground truth dot maps.
out_dir: output directory.
label_filepath: dataset labels text file containing categorical labels
'''

if __name__=="__main__":
    ####################################################################################
    ## Configuration for JHU++ - Test
    ####################################################################################
    '''
    data_dir = './eval/jhu_custom_topo1_patch100_topocount_test'; # contains prediction files
    root = './datasets/jhu/jhu_crowd_v2.0' # dataset root
    gt_dir = os.path.join(root, 'test','ground-truth_dots')  # ground truth dot maps
    label_filepath = os.path.join(root, 'test','image_labels.txt')  # labels file
    out_dir= './eval/jhu_custom_topo1_patch100_topocount_test'; # output directory
    log_filename = 'out_jhu_categories.txt'
    thresh_low = 0.4
    thresh_high = 0.5
    size_thresh = -1 # if set gets rid of connected components < size_thresh pixels
    '''
    ####################################################################################
    ## Configuration for JHU++ - Validation
    ####################################################################################
    #'''
    data_dir = './eval/jhu_custom_topo1_patch100_topocount_val'; # contains prediction files
    root = './datasets/jhu/jhu_crowd_v2.0' # dataset root
    gt_dir = os.path.join(root, 'val','ground-truth_dots')  # ground truth dot maps
    label_filepath = os.path.join(root, 'val','image_labels.txt')  # labels file
    out_dir= './eval/jhu_custom_topo1_patch100_topocount_val'; # output directory
    log_filename = 'out_jhu_categories.txt'
    thresh_low = 0.4
    thresh_high = 0.5
    size_thresh = -1 # if set gets rid of connected components < size_thresh pixels
    #'''

    ####################################################################################
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    with open(os.path.join(out_dir, log_filename), 'a+') as log_file:
        mae = 0
        rmse = 0
        files_count = 0
        cat_dict = {'low':0, 'medium':1, 'high':2, 'weather':3}
        mae_cat = np.array([0, 0, 0, 0]) # low, medium, high, weather
        rmse_cat = np.array([0, 0, 0, 0]) # low, medium, high, weather
        files_count_cat = np.array([0, 0, 0, 0]) # low, medium, high, weather

        # get prediction files paths
        e_soft_map_files = glob.glob(os.path.join(data_dir, '*_likelihood'+'.npy')) 
        print('files count', len(e_soft_map_files))

        # load labels file
        labels = np.loadtxt(label_filepath, dtype=str,delimiter=',')
        
        i=-1
        for file in e_soft_map_files:
            files_count += 1
            i +=1
            print('processing ', i)
            img_name = os.path.basename(file)[:-len('_likelihood.npy')]
            #g_dot=np.load(os.path.join(gt_dir, img_name + '_gt_dots.npy'))
            g_dot=np.load(os.path.join(gt_dir, img_name + '.npy'))
            g_count = g_dot.sum()
            e_soft = np.load(file)
            print('img_name',img_name)
            g_dot = g_dot[:e_soft.shape[0],:e_soft.shape[1]]
            label_row = labels[np.where(labels[:,0]==os.path.splitext(img_name)[0])].squeeze()
            print('label_row',label_row)            
            #print('g_dot',g_dot.shape)

            # get topological map from likelihood prediction
            e_hard = filters.apply_hysteresis_threshold(e_soft, thresh_low, thresh_high)
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
            e_dot = np.zeros(g_dot.shape)
            e_dot_vis = np.zeros(g_dot.shape)
            contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            e_coord_y = []
            e_coord_x = []
            for idx in range(len(contours)):
                #print('idx=',idx)
                contour_i = contours[idx]
                M = cv2.moments(contour_i)
                #print(M)
                if(M['m00'] == 0):
                    continue;
                cx = round(M['m10'] / M['m00'])
                cy = round(M['m01'] / M['m00'])
                e_dot_vis[cy-1:cy+1, cx-1:cx+1] = 1
                e_dot[cy, cx] = 1
                e_coord_y.append(cy)
                e_coord_x.append(cx)

            err= e_count - g_count
            mae += abs(err)
            rmse += err**2
            #print(img_name, e_count, g_count, err)
            #print(img_name, e_count, g_count,s_count, err)
            log_file.write("image {} e_count {} g_count {} err {} \n".format(img_name, e_count, g_count, err))
            log_file.flush()

            if(float(label_row[1]) <51):
                mae_cat[cat_dict['low']] = mae_cat[cat_dict['low']] + abs(err)
                rmse_cat[cat_dict['low']] = rmse_cat[cat_dict['low']] + err**2
                files_count_cat[cat_dict['low']] = files_count_cat[cat_dict['low']] + 1
            elif(float(label_row[1]) <501):
                mae_cat[cat_dict['medium']] = mae_cat[cat_dict['medium']] + abs(err)
                rmse_cat[cat_dict['medium']] = rmse_cat[cat_dict['medium']] + err**2
                files_count_cat[cat_dict['medium']] = files_count_cat[cat_dict['medium']] + 1
            else:
                mae_cat[cat_dict['high']] = mae_cat[cat_dict['high']] + abs(err)
                rmse_cat[cat_dict['high']] = rmse_cat[cat_dict['high']] + err**2
                files_count_cat[cat_dict['high']] = files_count_cat[cat_dict['high']] + 1
            if(int(label_row[3]) >0):
                mae_cat[cat_dict['weather']] = mae_cat[cat_dict['weather']] + abs(err)
                rmse_cat[cat_dict['weather']] = rmse_cat[cat_dict['weather']] + err**2
                files_count_cat[cat_dict['weather']] = files_count_cat[cat_dict['weather']] + 1

           
        mae /= files_count
        rmse = math.sqrt(rmse/files_count)
        #print('mae', mae, 'rmse', rmse)
        log_file.write("mae {} rmse {} \n".format(mae, rmse))
        log_file.flush()

        mae_cat = mae_cat/files_count_cat
        rmse_cat = np.sqrt(rmse_cat/files_count_cat)

        for cat in cat_dict.keys():
            #print('cat', cat, 'mae', mae_cat[cat_dict[cat]], 'rmse', rmse_cat[cat_dict[cat]], 'files_count', files_count_cat[cat_dict[cat]])
            log_file.write("cat {} mae {} rmse {} files_count {} \n".format(cat, mae_cat[cat_dict[cat]], rmse_cat[cat_dict[cat]], files_count_cat[cat_dict[cat]]))
            log_file.flush()
        sys.stdout.flush();


    print('Done.')
    print('Check output in: ', os.path.join(out_dir, log_filename))
