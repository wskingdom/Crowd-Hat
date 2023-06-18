import numpy as np
import scipy
import scipy.io as sio
import skimage.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
import PIL.Image as Image
import cv2


def gaussian_filter_density(img,points, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1):
    img_shape=[img.shape[0],img.shape[1]]
    print("Shape of image: ",img_shape,". Point count= ",len(points))
    density = np.zeros(img_shape, dtype=np.float32)
    density_std = np.zeros(img_shape, dtype=np.float32)
    density_nn = np.zeros(img_shape, dtype=np.float32)
    if(end_y <= 0):
        end_y = img.shape[0]
    if(end_x <= 0):
        end_x = img.shape[1]
    gt_count = len(points)
    if gt_count == 0:
        return density
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=2)
    
    max_sigma = 3.5; # kernel size = 7, kernel_width=15

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if(pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x):
            continue
        pt[1] -= start_y
        pt[0] -= start_x
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1])*0.125
            sigma = min(max_sigma, sigma)
        else:
            sigma = max_sigma;

        kernel_size = min(7, int(2*sigma + 0.5))
        sigma = kernel_size / 2
        kernel_width = kernel_size * 2 + 1
        #if(kernel_width < 15):
        #    print('i',i)
        #    print('distances',distances.shape)
        #    print('kernel_width',kernel_width)
        pnt_density = scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant',truncate=2)
        pnt_density /= pnt_density.sum()
        density_std[np.where(pnt_density > 0)] = sigma
        density_nn[np.where(pnt_density > 0)] = distances[i][1]
        density += pnt_density 
        

    density.astype(np.float16).dump(out_filepath)
    #density_std.astype(np.float16).dump(os.path.splitext(out_filepath)[0] + '_std.npy')
    #density_nn.astype(np.float16).dump(os.path.splitext(out_filepath)[0] + '_nn.npy')
    #density_hard = (density > 0).astype(np.uint8).dump(os.path.splitext(out_filepath)[0] + '_hard.npy')
    #io.imsave(out_filepath.replace('.npy', '.png'), (density/density.max()*255).astype(np.uint8))
    io.imsave(out_filepath.replace('.npy', '_solid.png'), ((density>0)*255).astype(np.uint8))
    print ('done.')
    return 

def process_dataset(images_dir, mat_dir, out_dir, scale, shorter_side=-1, longer_side=-1):
    img_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    for img_path in img_files:
        print(img_path)
        start_y=start_x=0
        end_y=end_x=-1
        img_basename = os.path.splitext(os.path.split(img_path)[1])[0]
        mat_filepath = os.path.join(mat_dir, img_basename + '_ann.mat')
        name_split = img_basename.split('_');
        out_gt_map_filepath =  os.path.join(out_dir, img_basename + '.npy')
        if(os.path.isfile(out_gt_map_filepath)):
            continue;
        img= Image.open(img_path)
        img= np.asarray(img)
        mat = sio.loadmat(mat_filepath)
        points = mat["annPoints"]
        if(shorter_side <0 and longer_side <0):
            img_scale = scale
        elif(longer_side >0):
            if(img.shape[0] > img.shape[1]):
                height = min(img.shape[0],longer_side)
                img_scale = height/img.shape[0]
            else:
                width = min(img.shape[1],longer_side)
                img_scale = width/img.shape[1]
        elif(shorter_side >0):
            if(img.shape[0] < img.shape[1]):
                height = min(img.shape[0],shorter_side)
                img_scale = height/img.shape[0]
            else:
                width = min(img.shape[1],shorter_side   )
                img_scale = width/img.shape[1]
        if(shorter_side >0 or longer_side >0):
            img2 = cv2.resize(img, (int(img.shape[1]*img_scale), int(img.shape[0]*img_scale)))
            io.imsave(os.path.join(out_dir_img,os.path.basename(img_path)), img2.astype(np.uint8))
        else:
            img2 = img
        points = (points * img_scale).astype(int)
        gaussian_filter_density(img,points, out_gt_map_filepath, start_y=start_y, start_x=start_x, end_y=end_y, end_x=end_x)

if __name__=="__main__":
    root = '../../datasets/UCF-QNRF_ECCV18/UCF-QNRF_ECCV18'

    # train dataset        
    images_dir = os.path.join(root,'Train')
    mat_dir = os.path.join(root,'Train')
    out_dir = os.path.join(root,'Train','gt_map_custom2_scalelong2048')    
    out_dir_img = os.path.join(root,'Train','img_scalelong2048')    
    scale = 1    
    shorter_side = -1
    longer_side = 2048

    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    if(not os.path.isdir(out_dir_img)):
        os.mkdir(out_dir_img)

    process_dataset(images_dir, mat_dir, out_dir, scale, shorter_side=shorter_side, longer_side=longer_side)

    ###############################################################################################
    # test dataset        
    images_dir = os.path.join(root,'Test')
    mat_dir = os.path.join(root,'Test')
    out_dir = os.path.join(root,'Test','gt_map_custom2_scaleshort2048')    
    out_dir_img = os.path.join(root,'Test','img_scaleshort2048')    
    scale = 1    
    shorter_side = 2048
    longer_side = -1

    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)
    if(not os.path.isdir(out_dir_img)):
        os.mkdir(out_dir_img)

    process_dataset(images_dir, mat_dir, out_dir, scale, shorter_side=shorter_side, longer_side=longer_side)




 