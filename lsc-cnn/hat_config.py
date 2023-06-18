import os

import torch


# vis = False
use_crowd_hat = False
prepare_training_data = False
# loc_test = False
default_thresh = 0.2
global_dic = {}
train_dic = {}
# img_root = r"/root/autodl-tmp/dataset/images"
# test_root = r"/root/autodl-tmp/dataset/test"

S = 64
L = 256
K = 4
alpha_ba = 1024
alpha_bc = 1

result_root = './results'
point_root = r'/home/ubuntu/dataset/nwpu/Points'
json_root = r'/home/ubuntu/dataset/nwpu/jsons'
# temp_root = r"/root/autodl-tmp/dataset/temp"
img_root = r'/home/ubuntu/dataset/nwpu/images'
nwpu_test = r'/home/ubuntu/dataset/test'
# json_root = r'/root/autodl-tmp/part_B_final/test_data/gt_count'
training_data_root = './training_data'
weights_root = './weights'
qnrf_img = r'/home/ubuntu/dataset/qnrf/jpegs'
qnrf_json = r'/home/ubuntu/dataset/qnrf/jsons'
sha_img = r'/root/autodl-tmp/dataset/sha_img'
sha_json = r'/root/autodl-tmp/dataset/sha_json'
shb_img = r'/root/autodl-tmp/dataset/shb_img'
shb_json = r'/root/autodl-tmp/dataset/shb_json'
vis_root = 'vis'

