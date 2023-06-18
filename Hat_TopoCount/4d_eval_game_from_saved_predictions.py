import numpy as np
import skimage.io as io
import os
import sys
import glob
import cv2
from skimage import filters
from skimage.measure import label
import math
from skimage.transform import resize

'''
Calculate Grid Average Mean absolute Error (GAME) localization metric using levels 0 through 5

R. Guerrero-Gomez-Olmedo et al., Extremely Overlapping Vehicle Counting, In Pattern Recognition and Image Analysis 2015.

Output files:
out_game.txt: per image and overall game scores for all levels

In main modify the directories:  
data_dir: path contains prediction files.
root: dataset root.
gt_dir: ground truth dot maps.
out_dir: output directory.
'''

if __name__=="__main__":

    ####################################################################################
    ## Configuration for ShanghaiTech Part B - Test
    ####################################################################################
    data_dir = './eval/sh_partb_custom_topo1_patch50_topocount_test'; # contains prediction files
    root = './datasets/ShanghaiTech/' # dataset root
    gt_dir = os.path.join(root,'part_B/test_data','ground-truth_dots') # ground truth dot maps
    out_dir= './eval/sh_partb_custom_topo1_patch50_topocount_test'; # output directory
    log_filename = 'out_game.txt'
    thresh_low = 0.4
    thresh_high = 0.5
    size_thresh = -1 # if set gets rid of connected components < size_thresh pixels

    ####################################################################################
    l = [0,1,2,3,4,5]
    mae=[0,0,0,0,0,0]
    rmse=[0,0,0,0,0,0]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    with open(os.path.join(out_dir, log_filename), 'a+') as log_file:
        # get prediction files paths
        e_soft_map_files = glob.glob(os.path.join(data_dir, '*_likelihood'+'.npy')) 
        #fig, axes = plt.subplots()

        i = -1
        for file in e_soft_map_files:
            i +=1
            img_name = os.path.basename(file)[:-len('_likelihood.npy')]
            #g_dot=np.load(os.path.join(gt_dir, img_name + '_gt_dots.npy'))
            g_dot=np.load(os.path.join(gt_dir, img_name + '.npy'))
            g_count = g_dot.sum()
            e_soft = np.load(file)
            e_soft = resize(e_soft.astype(np.float), (g_dot.shape[0], g_dot.shape[1]), anti_aliasing=True,preserve_range=True)
            #print('g_dot',g_dot.shape)
            #print('e_soft',e_soft.shape)

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
            for idx in range(len(contours)):
                contour_i = contours[idx]
                M = cv2.moments(contour_i)
                if(M['m00'] == 0):
                    continue;
                cx = round(M['m10'] / M['m00'])
                cy = round(M['m01'] / M['m00'])
                e_dot_vis[cy-1:cy+1, cx-1:cx+1] = 1
                e_dot[cy, cx] = 1

            err= e_count - g_count

            for indxl in range(len(l)):
                gdiv = l[indxl]
                #cells = 4**gdiv
                cells = 2**gdiv
                g_stride_y = g_dot.shape[0]//(cells)
                g_stride_x = g_dot.shape[1]//(cells)
                e_stride_y = e_dot.shape[0]//(cells)
                e_stride_x = e_dot.shape[1]//(cells)

                print('l',gdiv,'g_stride', g_stride_y,g_stride_x, 'e_stride',e_stride_y,e_stride_x)
                log_file.write("img_name {} level {} g_stride {} {} e_stride {} {} \n".format(img_name, gdiv,g_stride_y,g_stride_x, e_stride_y,e_stride_x))
                log_file.flush()
                sys.stdout.flush();
                for y in range(0,cells):
                    for x in range(0,cells):
                        e_cell_count = e_dot[y*e_stride_y:y*e_stride_y+e_stride_y, x*e_stride_x:x*e_stride_x+e_stride_x].sum()
                        g_cell_count = g_dot[y*g_stride_y:y*g_stride_y+g_stride_y, x*g_stride_x:x*g_stride_x+g_stride_x].sum()
                        mae[indxl] += abs(e_cell_count - g_cell_count)
                        rmse[indxl] += (e_cell_count - g_cell_count)**2
        

        for indxl in range(len(l)):
            gdiv = l[indxl]
            mae[indxl] = mae[indxl]/len(e_soft_map_files)
            rmse[indxl] = math.sqrt(rmse[indxl]/len(e_soft_map_files))
            print('game', gdiv, 'mae',mae[indxl], 'rmse', rmse[indxl])
            log_file.write("game {} mae {} rmse {} \n".format(gdiv, mae[indxl], rmse[indxl]))
            log_file.flush()
            sys.stdout.flush();

    print('Done.')
    print('Check output in: ', os.path.join(out_dir, log_filename))
