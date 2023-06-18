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
import json;

def gaussian_filter_density(img,points, boxes, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1):
    img_shape=[img.shape[0],img.shape[1]]
    print("Shape of image: ",img_shape,". Point count= ",len(points))
    density = np.zeros(img_shape, dtype=np.float32)
    density_std = np.zeros(img_shape, dtype=np.float32)
    density_nn = np.zeros(img_shape, dtype=np.float32)
    if(end_y <= 0):
        end_y = img.shape[0]
    if(end_x <= 0):
        end_x = img.shape[1]
    gt_count = points.shape[0]
    if(gt_count != 0):
        if(len(points.shape) < 2):
            points = points.reshape((1,-1))
        gt_count = points.shape[0]
    if gt_count > 1:
        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(points, k=2)
    
    max_sigma = 3.5; # kernel size = 7, kernel_width=15

    if(gt_count > 0):
        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            x,y = pt
            xmin, ymin, xmax, ymax = boxes[i]
            w = xmax - xmin
            h = ymax - ymin
            if(pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x):
                continue
            pt[1] -= start_y
            pt[0] -= start_x
            if int(pt[1])<img_shape[0] and int(pt[0])<img_shape[1]:
                pt2d[int(pt[1]),int(pt[0])] = 1.
            else:
                continue
            pt_max_sigma = max(max_sigma, w/4, h/4)
            if gt_count > 1:
                sigma = (distances[i][1])*0.125
                sigma = min(pt_max_sigma, sigma)
            else:
                sigma = pt_max_sigma;

            kernel_size = min(pt_max_sigma*2, int(2*sigma ))
            sigma = kernel_size / 2
            kernel_width = kernel_size * 2 + 1
            pnt_density = scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant',truncate=2)
            pnt_density /= pnt_density.sum()
            density_std[np.where(pnt_density > 0)] = sigma
            if(gt_count > 1):
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

def process_dataset(images_dir, json_dir, out_dir, scale):
    img_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    for img_path in img_files:
        print(img_path)
        start_y=start_x=0
        end_y=end_x=-1
        img_basename = os.path.splitext(os.path.split(img_path)[1])[0]
        json_filepath = os.path.join(jsons_dir, img_basename + '.json')
        name_split = img_basename.split('_');
        if(len(name_split) >9  ):
            start_y = int(name_split[3])
            start_x = int(name_split[5])
            end_y = start_y + int(name_split[7])
            end_x = start_x + int(name_split[9])
        out_gt_map_filepath =  os.path.join(out_dir, img_basename + '.npy')
        if(os.path.isfile(out_gt_map_filepath)):
            continue;
        if(not os.path.isfile(json_filepath)): # test data does not have annotation
            continue;
        img= Image.open(img_path)
        img= np.asarray(img)
        json_file = open(json_filepath, 'r');
        line = json_file.readline()
        line_json = json.loads(line)
        points=np.array(line_json['points'])
        boxes = np.array(line_json['boxes'])
        print('points',points.shape)
        print('boxes',boxes.shape)
        gaussian_filter_density(img,points, boxes, out_gt_map_filepath, start_y=start_y, start_x=start_x, end_y=end_y, end_x=end_x)

if __name__=="__main__":
    root = '../../datasets/nwpu-crowd'

    images_dir = os.path.join(root,'images')
    jsons_dir = os.path.join(root,'jsons')
    out_dir = os.path.join(root,'gt_map_custom2_boxes')    
    scale = 1    

    if(not os.path.isdir(out_dir)):
        os.mkdir(out_dir)

    process_dataset(images_dir, jsons_dir, out_dir, scale)

