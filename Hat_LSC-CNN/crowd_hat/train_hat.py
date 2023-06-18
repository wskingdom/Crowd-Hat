import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import cv2
import torch
import numpy as np
from PIL import Image
import hat_utils as utils
import data_augmentation as da
import hat_config as cfg
from utils.model import LSCCNN
import torchvision
from tqdm import tqdm

train_da = da.Compose([da.ToTensor(), da.RandomCrop(), da.RandomHorizontalFlip()])
test_da = da.Compose([da.ToTensor()])


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def prepare_training_data(img_root,json_root):
    """
    use the detection pipeline to do inference over the training dataset and save the data for training crowd hat network

    need to save the following data:

    Input Feature
        feature2d: including the 2D compressed matrices from area sizes and confidence scores, etc.
        feature1d: including the 1D compressed vectors from area sizes and confidence scores.
        feature2d_local: split the feature2d into K*K patches corresponding to K*K regions of the input image

    Output Value
        region_nms_thresholds: K*K optimal NMS threshold for K*K regions of the input image, labeled by linear search and F1-measure evaluation
        count: the ground truth crowd count over the image
    """
    save_root = cfg.training_data_root
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    checkpoint_path = '/home/ubuntu/lsc-cnn/models/train2/snapshots/scale_4_epoch_16.pth'
    # checkpoint_path = '/home/ubuntu/lsc/weights/scale_4_epoch_46.pth'
    model = LSCCNN()
    checkpoint = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(checkpoint)
    model.cuda().eval()
    img_list = sorted([name for name in os.listdir(img_root)])
    # pos = img_list.index('nwpu_2175.jpeg')
    # img_list = img_list[pos:]
    print(len(img_list))
    mae = 0
    mse = 0
    cnt = 0
    for name in img_list:
        # print(name)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        img_path = os.path.join(img_root, name)
        json_path = os.path.join(json_root, name.split('.')[0]+'.json')
        with open(json_path, 'r')as json_file:
            json_file = json.load(json_file)
        points = json_file['points']
        points = torch.tensor(points, dtype=torch.float32)
        json_file['points'] = points
        img = Image.open(img_path)
        img_list = [(img, json_file)]
        # transforms = test_da if int(name.split('.')[0].split('_')[1])>3109 else train_da
        # transforms = test_da
        transforms = train_da
        img_list = transforms(img_list)
        idx = 0
        for (img, target) in img_list:
            cfg.global_dic = {}
            cfg.train_dic = {}
            save_name = target['id'] + '_' + str(idx)
            cfg.global_dic['name'] = save_name
            gt_count = target['human_num']
            cfg.global_dic['count'] = gt_count
            img = torchvision.transforms.ToPILImage()(img)
            image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            image,ratio = utils.resize_image(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            target['points'] = target['points'] * ratio
            cfg.global_dic['points'] = (target['points']).tolist()
            utils.gt_distribution(cfg.global_dic['points'], cfg.global_dic['raw_img_size'])
            pred_dot_map, pred_box_map, img_out, count = model.predict_single_image(image, nms_thresh=0.2)
            mae += abs(gt_count - count)
            mse += (gt_count - count) ** 2
            cnt += 1
            idx += 1
            print(save_name, gt_count, count, mae / cnt, (mse / cnt) ** 0.5)
            utils.save2disk(os.path.join(save_root, save_name + '.json'))



if __name__ == '__main__':
    prepare_training_data(cfg.img_root,cfg.json_root)
    # prepare_training_data(cfg.qnrf_img,cfg.qnrf_json)
    # train_count_decoder(n=1, epochs=150)
    # train_nms_decoder(n=1, epochs=150)

