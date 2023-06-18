import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
from skimage import io
import cv2
import sys
from skimage.measure import label
from skimage import filters
import math
from PIL import Image
from tqdm import tqdm as tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from unet_vgg4_cc import UnetVggCC
import matplotlib.pyplot as plt
import hat_config as cfg
from hat_utils import compression, compression1d, compression2d
from my_dataset_test import CrowdDataset

if True:
    from count_decoder import CountDecoder

    count_model = CountDecoder()
    count_model.load_state_dict(
        torch.load(os.path.join(cfg.weights_root, 'count_decoder_2_54.9770.pth'), map_location='cpu'))
    count_model.cuda().eval()


def build_model(model_filename='topocount_nwpu.pth'):
    models_root_dir = '../pretrained_models'
    dropout_keep_prob = 1.0
    initial_pad = 126
    interpolate = 'False'
    conv_init = 'he'
    n_classes = 1
    n_channels = 1
    model = UnetVggCC(
        kwargs={'dropout_keep_prob': dropout_keep_prob, 'initial_pad': initial_pad, 'interpolate': interpolate,
                'conv_init': conv_init, 'n_classes': n_classes, 'n_channels': n_channels})
    model.load_state_dict(torch.load(os.path.join(models_root_dir, model_filename)), strict=True)
    model.cuda()
    model.eval()
    return model


def process_image(img, normalize=False):
    if type(img) is str:
        img = plt.imread(img) / 255
    elif type(img) is Image.Image:
        img = np.array(img) / 255
    h, w = img.shape[:2]
    max_size = 6400
    if h >= max_size or w >= max_size:
        ratio = min(max_size / h, max_size / w)
        w, h = int(w * ratio), int(h * ratio)
        img = cv2.resize(img, (w, h))
        # print('resize', w, h)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), 2)
    img = img[:, :, 0:3]
    max_scale = 16
    if max_scale > 1:
        ds_rows = int(img.shape[0] // max_scale) * max_scale
        ds_cols = int(img.shape[1] // max_scale) * max_scale
        pad_y1 = 0
        pad_y2 = 0
        pad_x1 = 0
        pad_x2 = 0
        if (ds_rows < img.shape[0]):
            pad_y1 = (max_scale - (img.shape[0] - ds_rows)) // 2
            pad_y2 = (max_scale - (img.shape[0] - ds_rows)) - pad_y1
        if (ds_cols < img.shape[1]):
            pad_x1 = (max_scale - (img.shape[1] - ds_cols)) // 2
            pad_x2 = (max_scale - (img.shape[1] - ds_cols)) - pad_x1
        img = np.pad(img, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), 'constant',
                     constant_values=(1,))
    new_h, new_w = img.shape[:2]
    img = img.transpose((2, 0, 1))
    img_tensor = torch.tensor(img, dtype=torch.float)
    if normalize:
        img_tensor = transforms.functional.normalize(img_tensor, mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])
    img_tensor = img_tensor.unsqueeze(dim=0)
    h_ratio, w_ratio = new_h / h, new_w / w
    cfg.global_dic['raw_img_size'] = [new_h, new_w]
    return img_tensor, h_ratio, w_ratio


def detect(img_tensor, model):
    criterion_sig = nn.Sigmoid()  # initialize sigmoid layer
    thresh_low = 0.4
    thresh_high = 0.5
    size_thresh = -1
    if (img_tensor.shape[-1] < 2048 and img_tensor.shape[-2] < 2048):
        img_tensor = img_tensor.cuda()
        # forward propagation
        et_dmap = model(img_tensor)[:, :, 2:-2, 2:-2]
        et_sig = criterion_sig(et_dmap.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap = et_dmap.squeeze().detach().cpu().numpy()
    elif (img_tensor.shape[-2] * img_tensor.shape[-1] > 30000):  # divide image into 3x3 overlapping to fit in GPU
        et_dmap = 0
        et_sig = np.zeros((img_tensor.shape[-2], img_tensor.shape[-1]))
        et_dmap = np.zeros((img_tensor.shape[-2], img_tensor.shape[-1]))
        y_part = img_tensor.shape[-2] // 3
        x_part = img_tensor.shape[-1] // 3
        overlap_y = 96
        overlap_x = 96
        max_scale = 16
        ds_rows = int(y_part // max_scale) * max_scale
        ds_cols = int(x_part // max_scale) * max_scale
        overlap_y += (max_scale - (y_part - ds_rows))
        overlap_x += (max_scale - (x_part - ds_cols))
        # print('#quad 0,0')
        # quad 0,0
        img_sub = img_tensor[:, :, :y_part + overlap_y, :x_part + overlap_x].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        et_sig[:y_part, :x_part] = et_sig_sub[:y_part, :x_part]
        et_dmap[:y_part, :x_part] = et_dmap_sub[:y_part, :x_part]
        del img_sub, et_dmap_sub, et_sig_sub
        # quad 0,1
        img_sub = img_tensor[:, :, :y_part + overlap_y, x_part - overlap_x // 2:2 * x_part + overlap_x // 2].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        et_sig[:y_part, x_part:2 * x_part] = et_sig_sub[:y_part, overlap_x // 2:overlap_x // 2 + x_part]
        et_dmap[:y_part, x_part:2 * x_part] = et_dmap_sub[:y_part, overlap_x // 2:overlap_x // 2 + x_part]
        del img_sub, et_dmap_sub, et_sig_sub
        # quad 0,2
        img_sub = img_tensor[:, :, :y_part + overlap_y, 2 * x_part - overlap_x:].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        et_sig[:y_part, -x_part:] = et_sig_sub[:y_part, -x_part:]
        et_dmap[:y_part, -x_part:] = et_dmap_sub[:y_part, -x_part:]
        del img_sub, et_dmap_sub, et_sig_sub
        # quad 1,0
        img_sub = img_tensor[:, :, y_part - overlap_y // 2:2 * y_part + overlap_y // 2, :x_part + overlap_x].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        et_sig[y_part:2 * y_part, :x_part] = et_sig_sub[overlap_y // 2:overlap_y // 2 + y_part, :x_part]
        et_dmap[y_part:2 * y_part, :x_part] = et_dmap_sub[overlap_y // 2:overlap_y // 2 + y_part, :x_part]
        del img_sub, et_dmap_sub, et_sig_sub
        # quad 1,1
        img_sub = img_tensor[:, :, y_part - overlap_y // 2:2 * y_part + overlap_y // 2,
                  x_part - overlap_x // 2:2 * x_part + overlap_x // 2].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        et_sig[y_part:2 * y_part, x_part:2 * x_part] = et_sig_sub[overlap_y // 2:overlap_y // 2 + y_part,
                                                       overlap_x // 2:overlap_x // 2 + x_part]
        et_dmap[y_part:2 * y_part, x_part:2 * x_part] = et_dmap_sub[overlap_y // 2:overlap_y // 2 + y_part,
                                                        overlap_x // 2:overlap_x // 2 + x_part]
        del img_sub, et_dmap_sub, et_sig_sub
        # quad 1,2
        img_sub = img_tensor[:, :, y_part - overlap_y // 2:2 * y_part + overlap_y // 2, 2 * x_part - overlap_x:].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        et_sig[y_part:2 * y_part, -x_part:] = et_sig_sub[overlap_y // 2:overlap_y // 2 + y_part, -x_part:]
        et_dmap[y_part:2 * y_part, -x_part:] = et_dmap_sub[overlap_y // 2:overlap_y // 2 + y_part, -x_part:]
        del img_sub, et_dmap_sub, et_sig_sub
        # quad 2,0
        img_sub = img_tensor[:, :, 2 * y_part - overlap_y:, :x_part + overlap_x].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        et_sig[-y_part:, :x_part] = et_sig_sub[-y_part:, :x_part]
        et_dmap[-y_part:, :x_part] = et_dmap_sub[-y_part:, :x_part]
        del img_sub, et_dmap_sub, et_sig_sub
        # quad 2,1
        img_sub = img_tensor[:, :, 2 * y_part - overlap_y:, x_part - overlap_x // 2:2 * x_part + overlap_x // 2].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        et_sig[-y_part:, x_part:2 * x_part] = et_sig_sub[-y_part:, overlap_x // 2:overlap_x // 2 + x_part]
        et_dmap[-y_part:, x_part:2 * x_part] = et_dmap_sub[-y_part:, overlap_x // 2:overlap_x // 2 + x_part]
        del img_sub, et_dmap_sub, et_sig_sub
        # quad 2,2
        img_sub = img_tensor[:, :, 2 * y_part - overlap_y:, 2 * x_part - overlap_x:].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        et_sig[-y_part:, -x_part:] = et_sig_sub[-y_part:, -x_part:]
        et_dmap[-y_part:, -x_part:] = et_dmap_sub[-y_part:, -x_part:]
        del img_sub, et_dmap_sub, et_sig_sub
    else:  # divide image into 2x2 overlapping to fit in GPU
        et_dmap = 0
        et_sig = np.zeros((img_tensor.shape[-2], img_tensor.shape[-1]))
        et_dmap = np.zeros((img_tensor.shape[-2], img_tensor.shape[-1]))
        y_half = img_tensor.shape[-2] // 2
        x_half = img_tensor.shape[-1] // 2
        overlap_y = 96
        overlap_x = 96
        max_scale = 16
        ds_rows = int(img_tensor.shape[-2] // max_scale) * max_scale
        ds_cols = int(img_tensor.shape[-1] // max_scale) * max_scale
        overlap_y += (max_scale - (img_tensor.shape[-2] - ds_rows))
        overlap_x += (max_scale - (img_tensor.shape[-1] - ds_cols))
        # print('#quad 0,0')
        # print('mem',torch.cuda.memory_allocated(device))
        # quad 0,0
        img_sub = img_tensor[:, :, :y_half + overlap_y, :x_half + overlap_x].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        img_sub = img_sub.squeeze().detach().cpu().numpy()
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print(':x_half',x_half)
        if (et_sig_sub.shape[0] > y_half + overlap_y):
            et_sig_sub = et_sig_sub[(et_sig_sub.shape[0] - y_half - overlap_y) // 2:-(
                    et_sig_sub.shape[0] - y_half - overlap_y) // 2, :]
            et_dmap_sub = et_dmap_sub[(et_dmap_sub.shape[0] - y_half - overlap_y) // 2:-(
                    et_dmap_sub.shape[0] - y_half - overlap_y) // 2, :]
        if (et_sig_sub.shape[1] > x_half + overlap_x):
            et_sig_sub = et_sig_sub[:, (et_sig_sub.shape[1] - x_half - overlap_x) // 2:-(
                    et_sig_sub.shape[1] - x_half - overlap_x) // 2]
            et_dmap_sub = et_dmap_sub[:, (et_dmap_sub.shape[1] - x_half - overlap_x) // 2:-(
                    et_dmap_sub.shape[1] - x_half - overlap_x) // 2]
        et_sig[:y_half, :x_half] = et_sig_sub[:y_half, :x_half]
        et_dmap[:y_half, :x_half] = et_dmap_sub[:y_half, :x_half]
        del img_sub, et_dmap_sub, et_sig_sub
        torch.cuda.empty_cache()
        # print('#quad 0,1')
        # print('mem',torch.cuda.memory_allocated(device))
        # quad 0,1
        img_sub = img_tensor[:, :, :y_half + overlap_y, x_half - overlap_x:].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print(':y_half',y_half)
        # print('x_half:',x_half, overlap_x)
        if (et_sig_sub.shape[0] > y_half + overlap_y):
            et_sig_sub = et_sig_sub[(et_sig_sub.shape[0] - y_half - overlap_y) // 2:-(
                    et_sig_sub.shape[0] - y_half - overlap_y) // 2, :]
            et_dmap_sub = et_dmap_sub[(et_dmap_sub.shape[0] - y_half - overlap_y) // 2:-(
                    et_dmap_sub.shape[0] - y_half - overlap_y) // 2, :]
        if (et_sig_sub.shape[1] > x_half + overlap_x):
            et_sig_sub = et_sig_sub[:, (et_sig_sub.shape[1] - x_half - overlap_x) // 2:-(
                    et_sig_sub.shape[1] - x_half - overlap_x) // 2]
            et_dmap_sub = et_dmap_sub[:, (et_dmap_sub.shape[1] - x_half - overlap_x) // 2:-(
                    et_dmap_sub.shape[1] - x_half - overlap_x) // 2]
        et_sig[:y_half, x_half:] = et_sig_sub[:y_half, overlap_y:]
        et_dmap[:y_half, x_half:] = et_dmap_sub[:y_half, overlap_y:]
        del img_sub, et_dmap_sub, et_sig_sub
        torch.cuda.empty_cache()
        # print('#quad 1,0')
        # print('mem',torch.cuda.memory_allocated(device))
        # quad 1,0
        img_sub = img_tensor[:, :, y_half - overlap_y:, :x_half + overlap_x].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print('y_half:',y_half, overlap_y)
        # print(':x_half',x_half)
        if (et_sig_sub.shape[0] > y_half + overlap_y):
            et_sig_sub = et_sig_sub[(et_sig_sub.shape[0] - y_half - overlap_y) // 2:-(
                    et_sig_sub.shape[0] - y_half - overlap_y) // 2, :]
            et_dmap_sub = et_dmap_sub[(et_dmap_sub.shape[0] - y_half - overlap_y) // 2:-(
                    et_dmap_sub.shape[0] - y_half - overlap_y) // 2, :]
        if (et_sig_sub.shape[1] > x_half + overlap_x):
            et_sig_sub = et_sig_sub[:, (et_sig_sub.shape[1] - x_half - overlap_x) // 2:-(
                    et_sig_sub.shape[1] - x_half - overlap_x) // 2]
            et_dmap_sub = et_dmap_sub[:, (et_dmap_sub.shape[1] - x_half - overlap_x) // 2:-(
                    et_dmap_sub.shape[1] - x_half - overlap_x) // 2]
        et_sig[y_half:, :x_half] = et_sig_sub[overlap_y:, :x_half]
        et_dmap[y_half:, :x_half] = et_dmap_sub[overlap_y:, :x_half]
        del img_sub, et_dmap_sub, et_sig_sub
        torch.cuda.empty_cache()
        # print('#quad 1,1')
        # print('mem',torch.cuda.memory_allocated(device))
        # quad 1,1
        img_sub = img_tensor[:, :, y_half - overlap_y:, x_half - overlap_x:].cuda()
        # print('img_sub',img_sub.shape)
        # et_dmap_sub=model(img_sub)[:,:,2:-2,2:-2]
        et_dmap_sub = model(img_sub)
        # print('et_dmap_sub',et_dmap_sub.shape)
        et_sig_sub = criterion_sig(et_dmap_sub.squeeze(dim=1)).squeeze().detach().cpu().numpy()
        et_dmap_sub = et_dmap_sub.squeeze().detach().cpu().numpy()
        # print('et_sig_sub',et_sig_sub.shape)
        # print('y_half:',y_half, overlap_y)
        # print('x_half:',x_half, overlap_x)
        if (et_sig_sub.shape[0] > y_half + overlap_y):
            et_sig_sub = et_sig_sub[(et_sig_sub.shape[0] - y_half - overlap_y) // 2:-(
                    et_sig_sub.shape[0] - y_half - overlap_y) // 2, :]
            et_dmap_sub = et_dmap_sub[(et_dmap_sub.shape[0] - y_half - overlap_y) // 2:-(
                    et_dmap_sub.shape[0] - y_half - overlap_y) // 2, :]
        if (et_sig_sub.shape[1] > x_half + overlap_x):
            et_sig_sub = et_sig_sub[:, (et_sig_sub.shape[1] - x_half - overlap_x) // 2:-(
                    et_sig_sub.shape[1] - x_half - overlap_x) // 2]
            et_dmap_sub = et_dmap_sub[:, (et_dmap_sub.shape[1] - x_half - overlap_x) // 2:-(
                    et_dmap_sub.shape[1] - x_half - overlap_x) // 2]
        et_sig[y_half:, x_half:] = et_sig_sub[overlap_y:, overlap_x:]
        et_dmap[y_half:, x_half:] = et_dmap_sub[overlap_y:, overlap_x:]
        del img_sub, et_dmap_sub, et_sig_sub
        torch.cuda.empty_cache()
    # print('mem',torch.cuda.memory_allocated(device))
    img_tensor = img_tensor.detach().cpu().numpy()
    # print('mem',torch.cuda.memory_allocated(device))
    e_hard = filters.apply_hysteresis_threshold(et_sig, thresh_low, thresh_high)
    e_hard2 = (e_hard > 0).astype(np.uint8)
    comp_mask = label(e_hard2)
    e_count = comp_mask.max()
    s_count = 0

    if (size_thresh > 0):
        for c in range(1, comp_mask.max() + 1):
            s = (comp_mask == c).sum()
            if (s < size_thresh):
                e_count -= 1
                s_count += 1
    # get dot predictions from topological map (centers of connected components)
    e_dot = np.zeros(et_sig.shape)
    e_box_vis = np.zeros(et_sig.shape)
    e_dot_vis = np.zeros(et_sig.shape)

    contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    confidences = []
    x_ctr = []
    y_ctr = []
    areas = []
    for idx in range(len(contours)):
        contour_i = contours[idx]
        M = cv2.moments(contour_i)
        if (M['m00'] == 0):
            continue
        cx = round(M['m10'] / M['m00'])
        cy = round(M['m01'] / M['m00'])
        e_dot_vis[cy - 1:cy + 1, cx - 1:cx + 1] = 1
        e_dot[cy, cx] = 1

        x_ctr.append(cx)
        y_ctr.append(cy)

        # xmin = np.min(contour_i[:, 0, 0])
        # ymin = np.min(contour_i[:, 0, 1])
        # xmax = np.max(contour_i[:, 0, 0])
        # ymax = np.max(contour_i[:, 0, 1])
        # if (xmax - xmin) * (ymax - ymin) < size_thresh:
        #     continue

        a = contour_i[:, 0, 0]
        b = contour_i[:, 0, 1]
        confidence = et_sig[b, a]
        confidence = np.mean(confidence)
        # if confidence < 0.4:
        #     continue
        confidence = (confidence - thresh_low) / (1 - thresh_low)  # normalize to range [0, 1] for better generalization
        confidences.append(confidence)
        areas.append(len(a))
    # dot_count = int(np.sum(e_dot))
    # if dot_count != e_count:
    #     print('!!!!!!!', dot_count, e_count)
    if cfg.prepare_training_data or cfg.use_crowd_hat:
        dot_map = compression(e_dot)
        hard_map = compression(e_hard2)
        like_map = compression(et_sig)
        assert np.max(et_sig) <= 1
        scores = np.clip((et_sig - thresh_low), a_min=0, a_max=1) / (1 - thresh_low)
        score_map = compression(scores)
        cfg.global_dic['topological'] = [like_map, score_map, dot_map, hard_map]
        x_ctr = torch.tensor(x_ctr, dtype=torch.int32)
        y_ctr = torch.tensor(y_ctr, dtype=torch.int32)
        areas = torch.tensor(areas, dtype=torch.float32)
        confidences = torch.tensor(confidences, dtype=torch.float32)
        compression2d(x_ctr, y_ctr, areas, confidences, et_sig.shape)
        compression1d(areas, confidences, et_sig.shape)
    # print(confidences)
    hat_count = -1
    if cfg.use_crowd_hat:
        count2d = torch.tensor([[cfg.global_dic['feature2d'][-1]]], dtype=torch.float32).cuda()
        area1d = torch.tensor([[cfg.global_dic['feature1d'][0]]], dtype=torch.float32).cuda()
        hat_count = int(round(count_model(count2d, area1d, mode=3).item()))
    return e_count, hat_count


def evaluate_counting(img_root, json_root, model):
    cfg.use_crowd_hat = True
    id_std = [i for i in range(3110, 3610, 1)]
    id_std[59] = 3098
    img_list = ['nwpu_' + str(idx) + '.jpeg' for idx in id_std]
    print(len(img_list))
    mae = 0
    mse = 0
    cnt = 0
    for name in img_list:
        img_path = os.path.join(img_root, name)
        json_path = os.path.join(json_root, name.split('.')[0] + '.json')
        with open(json_path, 'r')as json_file:
            json_file = json.load(json_file)
        gt_count = json_file['count']
        img_tensor, _, _ = process_image(img_path)
        count, hat_count = detect(img_tensor, model)
        cnt += 1
        mae += abs(gt_count - hat_count)
        mse += (gt_count - hat_count) ** 2
        print(name, gt_count, hat_count, mae / cnt, (mse / cnt) ** 0.5)

@torch.no_grad()
def test_nwpu_counting(img_root,model):
    if not os.path.exists(cfg.result_root):
        os.mkdir(cfg.result_root)
    txt = open(os.path.join(cfg.result_root,'hat_topo_counting.txt'), 'w')
    cfg.use_crowd_hat = True
    # checkpoint_path = 'weights/scale_4_epoch_1.pth'
    id_std = [i for i in range(3610, 5110, 1)]
    for img_id in id_std:
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        img_path = os.path.join(img_root, str(img_id) + '.jpg')
        img_tensor, _, _ = process_image(img_path)
        count, hat_count = detect(img_tensor, model)
        print(img_id, count, hat_count)
        txt.write(str(img_id) + ' ' + str(hat_count) + '\n')
    txt.close()

if __name__ == "__main__":
    topo = build_model()
    evaluate_counting(cfg.img_root, cfg.json_root, topo)
    # test_nwpu_counting(cfg.nwpu_test,topo)
