#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import os
import numpy as np
from skimage import io;
import glob;
import cv2 ; 
import sys;
import scipy
from scipy.ndimage.filters import gaussian_filter
#from scipy.misc import imresize
#from scipy.ndimage.filters import convolve    
from skimage.measure import label
from skimage import filters
import math
#from skimage.transform import resize


'''
Use previously generated predictions to calculate the average precision and average recall using the method outline in the paper:
C. Liu et al., Recurrent Attentive Zooming for Joint Crowd Counting and Precise Localization, CVPR 2019.

Pass arguments: sigma, threshold
According to paper the possible for sigma are in {5, 20, 40} and for threshold in [0.5, 0.95] with a step of 0.05.

Output files:
out_raz-fscore_sigma*_thresh*.txt: Average Precision (AP) and Average Recall (AR) over all images for specified sigma and threshold. Also the per image precision and recall for specified sigma and threshold.
prec_sigma*_thresh*.npy: array of average precision for specified sigma and threshold
recall_sigma*_thresh*.npy: array of average precision for specified sigma and threshold

In main modify the directories:  
data_dir: path contains prediction files.
root: dataset root.
gt_dir: ground truth dot maps.
out_dir: output directory.
'''

if __name__=="__main__":
    if(len(sys.argv) < 2):
        print('Missing sigma in arguments. Possible values include: 5, 20, 40.')
        exit()
    sigma = float(sys.argv[1]);
    if(len(sys.argv) < 3):
        print('Missing distance threshold in arguments. Possible values include: 0.5, 0.55, ..., 0.95.')
        exit()
    thresh = float(sys.argv[2]);

    print('sigma',sigma)
    print('thresh',thresh)
    ####################################################################################
    ## Configuration for ShanghaiTech Part B - Test
    ####################################################################################
    data_dir = './eval/sh_partb_custom_topo1_patch50_topocount_test'; # contains prediction files
    root = './datasets/ShanghaiTech/' # dataset root
    gt_dir = os.path.join(root,'part_B/test_data','ground-truth_dots') # ground truth dot maps
    out_dir= './eval/sh_partb_custom_topo1_patch50_topocount_test'; # output directory
    log_filename = 'out_raz-fscore_sigma'+str(sigma)+'_thresh'+str(thresh)+'.txt'
    thresh_low = 0.4
    thresh_high = 0.5
    size_thresh = -1 # if set gets rid of connected components < size_thresh pixels

    ####################################################################################
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    with open(os.path.join(out_dir, log_filename), 'a+') as log_file:

        # get prediction files paths
        e_soft_map_files = glob.glob(os.path.join(data_dir, '*_likelihood'+'.npy')) 
        print('files count', len(e_soft_map_files))        

        mae = 0
        rmse = 0

        sigma_list=[sigma]
        sigma_thresh_list=[thresh]

        # get gaussian max value
        sigma_max = []
        tmp = np.zeros((5,5))
        tmp[2,2]=1
        for sigma in sigma_list:
            g = gaussian_filter(tmp, sigma, mode='constant')
            sigma_max.append(g.max())
    
        
        arr_tp=np.zeros((len(sigma_list), len(sigma_thresh_list)))
        arr_fp=np.zeros((len(sigma_list), len(sigma_thresh_list)))
        arr_fn=np.zeros((len(sigma_list), len(sigma_thresh_list)))

        arr_prec=np.zeros((len(sigma_list), len(sigma_thresh_list)))
        arr_recall=np.zeros((len(sigma_list), len(sigma_thresh_list)))

        i = -1
        for file in e_soft_map_files:
            i +=1
            print('processing ', i)
            img_name = os.path.basename(file)[:-len('_likelihood.npy')]
            #g_dot=np.load(os.path.join(gt_dir, img_name + '_gt_dots.npy'))
            g_dot=np.load(os.path.join(gt_dir, img_name + '.npy'))
            g_count = g_dot.sum()
            e_soft = np.load(file)
            print('img_name',img_name)
            g_dot = g_dot[:e_soft.shape[0],:e_soft.shape[1]]
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


            for s in range(len(sigma_list)):
                sigma = sigma_list[s]
                for t in range(len(sigma_thresh_list)):
                    thresh = sigma_thresh_list[t]
                    tp = 0
                    fp = 0
                    fn = 0

                    g_dot_remaining = g_dot.copy()
                    for e_indx in range(len(e_coord_y)):
                        # create a map that is all zeros except one at the current prediction center
                        e_indx_map = np.zeros(g_dot_remaining.shape)
                        e_indx_map[e_coord_y[e_indx], e_coord_x[e_indx]] = 1

                        # generate a gaussian centered at current prediction with current sigma
                        # the gaussia is unnormalized; that is its peak is 1. therefore we divide by the filter by its maximum value.
                        et_sigma = gaussian_filter(e_indx_map, sigma=sigma, mode='constant')/sigma_max[s]        
                        # element-wise multiply et_sigma with the gt point map
                        gt_sigma = (et_sigma * g_dot_remaining)
                        # get max value which corresponds to closest gt
                        g_closest_val = gt_sigma.max()

                        # find if true positive based on current threshold
                        if(g_closest_val < thresh):
                            fp += 1
                        else:
                            tp += 1
                            # exclude matched point from ground truth map so that it is not matched again.
                            g_y, g_x = np.where(gt_sigma == g_closest_val)
                            g_dot_remaining[g_y[0], g_x[0]] = 0

                    # false negatives are remaining dots in ground truth map that were not matched.
                    fn = g_count - tp
                    if(fn < 0):
                        fn = 0
                    arr_tp[s,t] = arr_tp[s,t] + tp
                    arr_fp[s,t] = arr_fp[s,t] + fp
                    arr_fn[s,t] = arr_fn[s,t] + fn
            
                    prec = tp / (tp + fp)
                    recall = tp/ (tp + fn)
                    arr_prec[s,t] = arr_prec[s,t] + prec
                    arr_recall[s,t] = arr_recall[s,t] + recall
            
                    #print(img_name, e_count, g_count,s_count, err, prec, recall)
                    log_file.write("sigma {} threshold {} image {} e_count {} g_count {} err {} prec {} recall {} \n".format(sigma, thresh, img_name, e_count, g_count, err, prec, recall))
                    log_file.flush()

            sys.stdout.flush()
        n = i+1
        print('n files', n)
        arr_prec =arr_prec/n
        arr_recall =arr_recall/n
        arr_prec.dump(os.path.join(out_dir, 'prec'+'_sigma' + str(sigma)+'_thresh' + str(thresh) + '.npy'))
        arr_recall.dump(os.path.join(out_dir, 'prec'+'_sigma' + str(sigma)+'_thresh' + str(thresh) + '.npy'))


        for s in range(len(sigma_list)):
            sigma = sigma_list[s]
            for t in range(len(sigma_thresh_list)):
                thresh = sigma_thresh_list[t]
                log_file.write("sigma {} threshold {} AP {} AR {} \n".format(sigma, thresh, arr_prec[s,t], arr_recall[s,t]))
                log_file.flush()

        sys.stdout.flush()

    print('Done.')
    print('Check output in: ', os.path.join(out_dir, log_filename))
