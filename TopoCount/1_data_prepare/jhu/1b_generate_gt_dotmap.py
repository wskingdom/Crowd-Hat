import numpy as np
import scipy
import scipy.io as sio
import skimage.io as io
from scipy.ndimage.filters import gaussian_filter
import os
import glob
#from matplotlib import pyplot as plt
#import h5py
import PIL.Image as Image
#from matplotlib import cm as CM


def gaussian_filter_density(img,points, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1):
    img_shape=[img.shape[0],img.shape[1]]
    print("Shape of image: ",img_shape,". Point count= ",len(points))
    density = np.zeros(img_shape, dtype=np.float32)
    if(end_y <= 0):
        end_y = img.shape[0]
    if(end_x <= 0):
        end_x = img.shape[1]
    gt_count = len(points)
    if gt_count == 0:
        return density

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        x,y,w,h,o,b = pt
        if(pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x):
            continue
        pt[1] -= start_y
        pt[0] -= start_x
        if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
            pt2d[int(pt[1]),int(pt[0])] = 1.
        else:
            continue
        if(int(pt[1]) >= density.shape[0]  or int(pt[0]) >= density.shape[1]):
            print('error:')
            print('img:', img.shape[0], ' - ', img.shape[1])
            print(points[i][1], ' - ', points[i][0])
            print(pt[1], ' - ' , pt[0])
            print(density.shape)
        if(int(pt[1]) >= density.shape[0]):
            pt[1] = density.shape[0] - 1
        if(int(pt[0]) >= density.shape[1]):
            pt[0] = density.shape[1] - 1
        density[int(pt[1]),int(pt[0])] = 1;

        density.astype(np.uint8).dump(out_filepath)

    #io.imsave(out_filepath.replace('.npy', '.png'), (density/density.max()*255).astype(np.uint8))
    io.imsave(out_filepath.replace('.npy', '_solid.png'), ((density>0)*255).astype(np.uint8))
    print ('done.')
    return 

def process_dataset(images_dir, txt_dir, out_dir, scale):
    img_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    for img_path in img_files:
        print(img_path)
        start_y=start_x=0
        end_y=end_x=-1
        img_basename = os.path.splitext(os.path.split(img_path)[1])[0]
        txt_filepath = os.path.join(txt_dir, img_basename + '.txt')
        out_gt_map_filepath =  os.path.join(out_dir, img_basename + '.npy')
        if(os.path.isfile(out_gt_map_filepath)):
            continue;
        img= Image.open(img_path)
        img= np.asarray(img)
        points = np.loadtxt(txt_filepath,dtype=int)
        print('points',points.shape)
        gaussian_filter_density(img,points, out_gt_map_filepath, start_y=start_y, start_x=start_x, end_y=end_y, end_x=end_x)

if __name__=="__main__":
    root = '../../datasets/jhu/jhu_crowd_v2.0'

    # train dataset        
    images_dir = os.path.join(root, 'train','images')
    txt_dir = os.path.join(root, 'train','gt')
    out_dir = os.path.join(root, 'train','ground-truth_dots')    
    scale = 1    

    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)

    process_dataset(images_dir, txt_dir, out_dir, scale)

    # val dataset        
    images_dir = os.path.join(root, 'val','images')
    txt_dir = os.path.join(root, 'val','gt')
    out_dir = os.path.join(root, 'val','ground-truth_dots')    
    scale = 1    

    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)

    process_dataset(images_dir, txt_dir, out_dir, scale)

    # test dataset        
    images_dir = os.path.join(root, 'test','images')
    txt_dir = os.path.join(root, 'test','gt')
    out_dir = os.path.join(root, 'test','ground-truth_dots')    
    scale = 1    

    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)

    process_dataset(images_dir, txt_dir, out_dir, scale)

