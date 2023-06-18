import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import cv2
from utils.model import LSCCNN
import torch
import hat_config as cfg
import hat_utils as utils
from thop import profile
from tqdm import tqdm
import numpy as np


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


@torch.no_grad()
def evaluate_counting(img_root, json_root, nms_threshold):
    cfg.prepare_train = False
    cfg.use_crowd_hat = True
    # checkpoint_path = 'weights/scale_4_epoch_1.pth'
    checkpoint_path = '/home/ubuntu/lsc-cnn/models/train2/snapshots/scale_4_epoch_16.pth'
    model = LSCCNN()
    checkpoint = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(checkpoint)
    model.cuda().eval()
    id_std = [i for i in range(3110, 3610, 1)]
    id_std[59] = 3098
    img_list = ['nwpu_' + str(idx) + '.jpeg' for idx in id_std]
    # img_list = sorted([name for name in os.listdir(img_root) if 'nwpu' in name and 3109<int(name.split('_')[1].split('.')[0])<=3609])
    print(len(img_list))
    mae = 0
    mse = 0
    cnt = 0
    for name in img_list:
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        img_path = os.path.join(img_root, name)
        json_path = os.path.join(json_root, name.split('.')[0] + '.json')
        with open(json_path, 'r')as json_file:
            json_file = json.load(json_file)
        gt_count = json_file['human_num']
        cfg.global_dic['count'] = gt_count
        image = cv2.imread(img_path)
        image, _ = utils.resize_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_dot_map, pred_box_map, img_out, count = model.predict_single_image(image, nms_thresh=nms_threshold)
        count = cfg.global_dic['hat_count']
        mae += abs(gt_count - count)
        mse += (gt_count - count) ** 2
        cnt += 1
        print(nms_threshold, name, gt_count, count, mae / cnt, (mse / cnt) ** 0.5)


@torch.no_grad()
def evaluate_localization(img_root, json_root, nms_threshold):
    cfg.prepare_train = False
    cfg.use_crowd_hat = True
    if not os.path.exists(cfg.result_root):
        os.mkdir(cfg.result_root)
    # checkpoint_path = 'weights/scale_4_epoch_1.pth'
    txt_before = open(os.path.join(cfg.result_root, 'localization_before.txt'), 'w')
    txt_after = open(os.path.join(cfg.result_root, 'localization_after.txt'), 'w')
    checkpoint_path = '/home/ubuntu/lsc-cnn/models/train2/snapshots/scale_4_epoch_16.pth'
    model = LSCCNN()
    checkpoint = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(checkpoint)
    model.cuda().eval()
    id_std = [i for i in range(3143, 3610, 1)]
    # id_std[59] = 3098
    img_list = ['nwpu_' + str(idx) + '.jpeg' for idx in id_std]
    # img_list = sorted([name for name in os.listdir(img_root) if 'nwpu' in name and 3109<int(name.split('_')[1].split('.')[0])<=3609])
    print(len(img_list))
    tp_before, fp_before, fn_before, tp_after, fp_after, fn_after = [], [], [], [], [], []
    cnt = 0
    for name in img_list:
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        img_path = os.path.join(img_root, name)
        json_path = os.path.join(json_root, name.split('_')[1].split('.')[0] + '.json')
        with open(json_path, 'r')as json_file:
            json_file = json.load(json_file)
        gt_count = json_file['human_num']
        gt_boxes = np.array(json_file['boxes'])
        sigma = np.empty(0)
        if gt_boxes.any():
            gt_boxes_w = gt_boxes[:, 2] - gt_boxes[:, 0]
            gt_boxes_h = gt_boxes[:, 3] - gt_boxes[:, 1]
            sigma = np.sqrt(gt_boxes_w * gt_boxes_w + gt_boxes_h * gt_boxes_h) / 2
        cfg.global_dic['count'] = gt_count
        image = cv2.imread(img_path)
        image, ratio = utils.resize_image(image)
        gt_p = np.rint(json_file['points']).astype('int')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_dot_map, pred_box_map, img_out, count = model.predict_single_image(image, nms_thresh=nms_threshold)

        boxes = cfg.global_dic['boxes'] / ratio
        x_ctr = (boxes[:, 0] + boxes[:, 2]) / 2
        y_ctr = (boxes[:, 1] + boxes[:, 3]) / 2
        pred_p = np.rint(np.stack((x_ctr, y_ctr), axis=1)).astype('int')
        tp, fp, fn = utils.evaluate_f1_with_box(pred_p, gt_p, sigma)
        tp_before.append(tp)
        fp_before.append(fp)
        fn_before.append(fn)
        txt_before.write(name.split('_')[1].split('.')[0] + ' ' + str(len(boxes)) + ' ')
        for point in pred_p:
            txt_before.write(str(point[0]) + ' ' + str(point[1]) + ' ')
        txt_before.write('\n')

        boxes = cfg.global_dic['hat_boxes'] / ratio
        x_ctr = (boxes[:, 0] + boxes[:, 2]) / 2
        y_ctr = (boxes[:, 1] + boxes[:, 3]) / 2
        pred_p = np.rint(np.stack((x_ctr, y_ctr), axis=1)).astype('int')
        tp, fp, fn = utils.evaluate_f1_with_box(pred_p, gt_p, sigma)
        tp_after.append(tp)
        fp_after.append(fp)
        fn_after.append(fn)
        txt_after.write(name.split('_')[1].split('.')[0] + ' ' + str(len(boxes)) + ' ')
        for point in pred_p:
            txt_after.write(str(point[0]) + ' ' + str(point[1]) + ' ')
        txt_after.write('\n')

        ap_before = np.sum(tp_before) / (np.sum(tp_before) + np.sum(fp_before) + 1e-20)
        ar_before = np.sum(tp_before) / (np.sum(tp_before) + np.sum(fn_before) + 1e-20)
        f1m_before = 2 * ap_before * ar_before / (ap_before + ar_before)

        ap_after = np.sum(tp_after) / (np.sum(tp_after) + np.sum(fp_after) + 1e-20)
        ar_after = np.sum(tp_after) / (np.sum(tp_after) + np.sum(fn_after) + 1e-20)
        f1m_after = 2 * ap_after * ar_after / (ap_after + ar_after)
        improvement = f1m_after - f1m_before
        print(nms_threshold, name, f1m_before, f1m_after, format(improvement * 100, '.2f') + '%')
    txt_before.close()
    txt_after.close()


@torch.no_grad()
def test_nwpu_counting(img_root, nms_threshold=0.2):
    cfg.prepare_train = False
    cfg.use_crowd_hat = True
    if not os.path.exists(cfg.result_root):
        os.mkdir(cfg.result_root)
    txt = open(os.path.join(cfg.result_root,'hat_lsc_counting.txt'), 'w')
    # checkpoint_path = 'weights/scale_4_epoch_1.pth'
    checkpoint_path = '/home/ubuntu/lsc-cnn/models/train2/snapshots/scale_4_epoch_16.pth'
    model = LSCCNN()
    checkpoint = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    id_std = [i for i in range(3610, 5110, 1)]
    for img_id in id_std:
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        img_path = os.path.join(img_root, str(img_id) + '.jpg')
        image = cv2.imread(img_path)
        image, _ = utils.resize_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_dot_map, pred_box_map, img_out, count = model.predict_single_image(image, nms_thresh=nms_threshold)
        count = cfg.global_dic['hat_count']
        print(img_id, count)
        txt.write(str(img_id) + ' ' + str(count) + '\n')
    txt.close()


@torch.no_grad()
def test_nwpu_localization(img_root, nms_threshold):
    cfg.prepare_train = False
    cfg.use_crowd_hat = True
    if not os.path.exists(cfg.result_root):
        os.mkdir(cfg.result_root)
    txt = open(os.path.join(cfg.result_root, 'hat_lsc_localization.txt'), 'w')
    checkpoint_path = '/home/ubuntu/lsc-cnn/models/train2/snapshots/scale_4_epoch_16.pth'
    model = LSCCNN()
    checkpoint = torch.load(checkpoint_path)['state_dict']
    model.load_state_dict(checkpoint)
    model.cuda().eval()
    id_std = [i for i in range(3610, 5110, 1)]
    for img_id in id_std:
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        img_path = os.path.join(img_root, str(img_id) + '.jpg')
        image = cv2.imread(img_path)
        image, ratio = utils.resize_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pred_dot_map, pred_box_map, img_out, count = model.predict_single_image(image, nms_thresh=nms_threshold)
        boxes = cfg.global_dic['boxes'] / ratio
        x_ctr = (boxes[:, 0] + boxes[:, 2]) / 2
        y_ctr = (boxes[:, 1] + boxes[:, 3]) / 2
        pred_p = np.rint(np.stack((x_ctr, y_ctr), axis=1)).astype('int')
        txt.write(str(img_id) + ' ' + str(len(boxes)) + ' ')
        for point in pred_p:
            txt.write(str(point[0]) + ' ' + str(point[1]) + ' ')
        txt.write('\n')
        print(img_id)
    txt.close()


if __name__ == '__main__':
    # evaluate_counting(cfg.img_root,cfg.json_root,0.2)
    evaluate_localization(cfg.img_root, cfg.json_root, 0.2)
    # test_nwpu_counting(cfg.nwpu_test)
