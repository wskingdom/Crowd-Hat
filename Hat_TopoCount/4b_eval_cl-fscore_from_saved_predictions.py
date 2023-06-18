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
from skimage.measure import label
from skimage import filters
import math
from skimage.transform import resize


'''
Use previously generated predictions to calculate the f-score using the method outline in the paper:
H. Idrees et al., Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds, ECCV 2018.

Output files:
out_cl-fscore.txt: the micro- and macro- averaged f-scores
*_e_centers.npy: numpy of predicted dots (centers of connected components).
*_e_centers.png: image visualization of predicted dots (centers of connected components).
precision.npy: array of overall precision per threshold. Array size = max_dist_thresh+1. Index 0 is not used.
recall.npy: array of overall recall per threshold. Array size = max_dist_thresh+1. Index 0 is not used.
f1.npy: array of overall fscore per threshold. Array size = max_dist_thresh+1. Index 0 is not used.
precision_img.npy: array of images precision per threshold. Array size = max_dist_thresh+1 x number of images. Index 0 is not used.
recall_img.npy: array of images recall per threshold. Array size = max_dist_thresh+1 x number of images. Index 0 is not used.
f1_img.npy: array of images fscore per threshold. Array size = max_dist_thresh+1 x number of images. Index 0 is not used.
image_names.npy: array of image names corresponding to stats in *_img.npy
image_names.txt: txt file comma separated list of image names corresponding to stats in *_img.npy

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
    log_filename = 'out_cl-fscore.txt'
    thresh_low = 0.4
    thresh_high = 0.5
    size_thresh = -1 # if set gets rid of connected components < size_thresh pixels

    ####################################################################################
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    with open(os.path.join(out_dir, log_filename), 'a+') as log_file:
        # get prediction files paths
        e_soft_map_files = glob.glob(os.path.join(data_dir, '*_likelihood'+'.npy')) 
        #fig, axes = plt.subplots()

        mae = 0
        rmse = 0
        k=20 # maximum number of nearest neigbors to consider

        max_dist_thresh = 100
        # the arrays for tp, fp, fn, precision, recall, and f1 only use the entries from 1 to max_dist_thresh. Do not use index 0.
        tp = np.zeros(max_dist_thresh+1) 
        fp = np.zeros(max_dist_thresh+1)
        fn = np.zeros(max_dist_thresh+1)
        # precision, recall, and f1 arrays will hold evaluation for each threshold over all images.
        precision = np.zeros(max_dist_thresh+1)
        recall = np.zeros(max_dist_thresh+1)
        f1 = np.zeros(max_dist_thresh+1)

        # precision_img, recall_img, and f1_img arrays will hold evaluation for each threshold per image. 
        precision_img=np.zeros((max_dist_thresh+1, len(e_soft_map_files)))
        recall_img =np.zeros((max_dist_thresh+1, len(e_soft_map_files)))
        f1_img =np.zeros((max_dist_thresh+1, len(e_soft_map_files)))

        image_names = ['']

        i = -1
        for file in e_soft_map_files:
            i +=1
            img_name = os.path.basename(file)[:-len('_likelihood.npy')]
            image_names.append(img_name)
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
            io.imsave(os.path.join(out_dir, img_name + '_e_centers_'+'.png'), (e_dot_vis*255).astype(np.uint8))
            e_dot.astype(np.uint8).dump(os.path.join(out_dir, img_name + '_e_centers_'+'.npy'))

            err= e_count - g_count
            mae += abs(err)
            rmse += err**2
            #print(img_name, e_count, g_count, err)
            #print(img_name, e_count, g_count,s_count, err)
            log_file.write("image {} e_count {} g_count {} err {} \n".format(img_name, e_count, g_count, err))
            log_file.flush()



            leafsize = 2048
            e_coords = np.where(e_dot > 0) # convert dot map to points list
            # build kdtree
            #z = zip(np.where(e_dot > 0)[1], np.where(e_dot > 0)[0])
            z = np.zeros((len(e_coords[0]),2))
            z[:,0] = e_coords[0]
            z[:,1] = e_coords[1]
            #print('z',z)
            tree = scipy.spatial.KDTree(z, leafsize=leafsize)
            print('tree.data.shape', tree.data.shape)

            for dist_thresh in range(1,max_dist_thresh+1):
                tp_img = 0
                fn_img = 0
                fp_img = 0

                e_dot_processing = np.copy(e_dot)
                gt_points = np.where(g_dot > 0)
                # for each point in ground truth find match in prediction
                for pi in range(len(gt_points[0])):
                    p = [[gt_points[0][pi], gt_points[1][pi]]]
                    #print('p',p)
                    distances, locations = tree.query(p, k=k,distance_upper_bound =dist_thresh)
                    #print('distances',distances)
                    #print('locations',locations)
                    match = False
                    # true positive, tp,  match to ground truth point is nearest neighbor prediction that was not matched to another gt point before
                    for nn in range(min(k,len(locations[0]))):
                        if((len(locations[0]) > 0) and (locations[0][nn] < tree.data.shape[0]) and (e_dot_processing[int(tree.data[locations[0][nn]][0]),int(tree.data[locations[0][nn]][1])] > 0)):
                            tp[dist_thresh] += 1
                            tp_img +=1
                            e_dot_processing[int(tree.data[locations[0][nn]][0]),int(tree.data[locations[0][nn]][1])] = 0
                            match = True
                            break
                    # false negative, fn,  no match is found to ground truth point
                    if(not match):
                        fn[dist_thresh] += 1
                        fn_img +=1

                # false positive, fp,  remaining points in prediction that were not matched to any point in ground truth
                fp[dist_thresh] += e_dot_processing.sum()
                fp_img +=e_dot_processing.sum()
                sys.stdout.flush();

                # calculate image precision, recall, and f-score for current threshold
                if(tp_img + fp_img == 0):
                    precision_img[dist_thresh, i] = 1
                else:
                    precision_img[dist_thresh, i] = tp_img/(tp_img + fp_img)
                if(tp_img + fn_img == 0):
                    recall_img[dist_thresh, i] = 1
                else:
                    recall_img[dist_thresh, i] = tp_img/(tp_img + fn_img) # True pos rate
                f1_img[dist_thresh, i] = 2*(( precision_img[dist_thresh, i]*recall_img[dist_thresh, i] )/( precision_img[dist_thresh, i]+recall_img[dist_thresh, i] ))

        #mae /= len(e_soft_map_files)
        #rmse = math.sqrt(rmse/len(e_soft_map_files))
        ##print('dist_thresh', dist_thresh, 'tp',tp[dist_thresh], 'fn',fn[dist_thresh], 'fp', fp[dist_thresh])

        #print('mae', mae)
        #print('rmse', rmse)

        # calculate overall precision, recall, and f-score for each threshold
        for dist_thresh in range(1,max_dist_thresh+1):
            precision[dist_thresh] = tp[dist_thresh]/(tp[dist_thresh] + fp[dist_thresh])
            recall[dist_thresh] = tp[dist_thresh]/(tp[dist_thresh] + fn[dist_thresh]) # True pos rate
            f1[dist_thresh] = 2*((precision[dist_thresh]*recall[dist_thresh])/(precision[dist_thresh]+recall[dist_thresh]))

            #print(dist_thresh, precision[dist_thresh], recall[dist_thresh],f1[dist_thresh])
            log_file.write("distance-threshold {} precision {} recall {} fscore {}  \n".format(dist_thresh, precision[dist_thresh], recall[dist_thresh],f1[dist_thresh]))
            log_file.flush()


        # get mean precision, recall, and f-score over all thresholds (micro average)
        #print('avg precision_overall',precision[1:max_dist_thresh].mean())
        #print('avg recall_overall', recall[1:max_dist_thresh].mean())
        #print('avg F1_overall', f1[1:max_dist_thresh].mean())

        # get mean precision, recall, and f-score over all thresholds and images (macro average)
        #print('avg precision_img', precision_img.mean(axis=-1))
        #print('avg recall_img', recall_img.mean(axis=-1))
        #print('avg f1_img', f1_img.mean(axis=-1))
        log_file.write("avg_precision_img {} \n".format(precision_img.mean(axis=-1)))
        log_file.write("avg_recall_img {} \n".format(recall_img.mean(axis=-1)))
        log_file.write("avg_F1_img {} \n".format(f1_img.mean(axis=-1)))
        log_file.flush()

        precision.dump(os.path.join(out_dir, 'precision.npy'))
        recall.dump(os.path.join(out_dir, 'recall.npy'))
        f1.dump(os.path.join(out_dir, 'f1.npy'))

        precision_img.dump(os.path.join(out_dir, 'precision_img.npy'))
        recall_img.dump(os.path.join(out_dir, 'recall_img.npy'))
        f1_img.dump(os.path.join(out_dir, 'f1_img.npy'))
        np.array(image_names).dump(os.path.join(out_dir, 'image_names.npy'))
        np.savetxt(os.path.join(out_dir, 'image_names.txt'), np.array(image_names).astype('U'), fmt='%s', delimiter=',')

        #log_file.write("\n")
        #log_file.write("micro overall average precision {} \n".format(precision[1:max_dist_thresh].mean()))
        #log_file.write("micro overall average recall {} \n".format(recall[1:max_dist_thresh].mean()))
        #log_file.write("micro overall average F1 {} \n".format(f1[1:max_dist_thresh].mean()))
        #log_file.flush()

        #log_file.write("\n")
        #log_file.write("macro image average precision {} \n".format(precision_img[1:max_dist_thresh].mean()))
        #log_file.write("macro image average recall {} \n".format(recall_img[1:max_dist_thresh].mean()))
        #log_file.write("macro image average F1 {} \n".format(f1_img[1:max_dist_thresh].mean()))
        #log_file.flush()

        log_file.write("\n")
        log_file.write("overall average precision {} \n".format(precision[1:max_dist_thresh].mean()))
        log_file.write("overall average recall {} \n".format(recall[1:max_dist_thresh].mean()))
        log_file.write("overall average F1 {} \n".format(f1[1:max_dist_thresh].mean()))
        log_file.flush()

    print('Done.')
    print('Check output in: ', os.path.join(out_dir, log_filename))
