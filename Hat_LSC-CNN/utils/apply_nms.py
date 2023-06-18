"""
apply_nms.py: Wrapper for nms.py
Authors : svp
"""
import os

from utils.nms import nms
import numpy as np
from crowd_hat.hat_utils import compression2d, compression1d, alignment_scheme, region_nms, restore_boxes
import hat_config as cfg
import torch

if True:
    from crowd_hat.count_decoder import CountDecoder
    from crowd_hat.nms_decoder import NMSDecoder

    count_model = CountDecoder()
    count_model.load_state_dict(
        torch.load(os.path.join(cfg.weights_root, 'count_decoder_2_74.7673.pth'), map_location='cpu'))
    count_model.cuda().eval()
    nms_model = NMSDecoder()
    nms_model.load_state_dict(
        torch.load(os.path.join(cfg.weights_root, 'nms_decoder_0.0153.pth'), map_location='cpu')
    )
    nms_model.cuda().eval()
'''
    Extracts confidence map and box map from N (N=4 here)
	channel input.
	
    Parameters:
    -----------
    confidence_map - (list) list of confidences for N channels
    hmap - (list) list of box values for N channels

    Returns
    -------
    nms_conf_map - (HXW) single channel confidence score map 
	nms_conf_box - (HXW) single channel box map.
'''


def extract_conf_points(confidence_map, hmap):
    nms_conf_map = np.zeros_like(confidence_map[0])
    nms_conf_box = np.zeros_like(confidence_map[0])

    idx_1 = np.where(np.logical_and(confidence_map[0] > 0, confidence_map[1] <= 0))
    idx_2 = np.where(np.logical_and(confidence_map[0] <= 0, confidence_map[1] > 0))
    idx_common = np.where(np.logical_and(confidence_map[0] > 0, confidence_map[1] > 0))

    nms_conf_map[idx_1] = confidence_map[0][idx_1]
    nms_conf_map[idx_2] = confidence_map[1][idx_2]

    nms_conf_box[idx_1] = hmap[0][idx_1]
    nms_conf_box[idx_2] = hmap[1][idx_2]

    for ii in range(len(idx_common[0])):
        x, y = idx_common[0][ii], idx_common[1][ii]
        if confidence_map[0][x, y] > confidence_map[1][x, y]:
            nms_conf_map[x, y] = confidence_map[0][x, y]
            nms_conf_box[x, y] = hmap[0][x, y]
        else:
            nms_conf_map[x, y] = confidence_map[1][x, y]
            nms_conf_box[x, y] = hmap[1][x, y]

    assert (np.sum(nms_conf_map > 0) == len(idx_1[0]) + len(idx_2[0]) + len(idx_common[0]))

    return nms_conf_map, nms_conf_box


'''
    Wrapper function to perform NMS
    
    Parameters:
    -----------
    confidence_map - (list) list of confidences for N channels
    hmap - (list) list of box values for N channels
    wmap - (list) list of box values for N channels	
    dotmap_pred_downscale -(int) prediction scale
    thresh - (float) Threshold for NMS.

    Returns
    -------
    x, y - (list) list of x-coordinates and y-coordinates to keep
           after NMS.
    h, w - (list) list of height and width of the corresponding x, y 
            points.
    scores - (list) list of confidence for h and w at (x, y) point.

'''


def chunk_confidence_map(e_map, size):
    maps = []
    h, w = e_map[0].shape
    x_unit = w / size
    y_unit = h / size
    x_range = np.ceil(np.arange(size + 1) * x_unit).astype('int32').tolist()
    y_range = np.ceil(np.arange(size + 1) * y_unit).astype('int32').tolist()
    for one_map in e_map:
        one_map = torch.from_numpy(one_map)
        ret_map = [[0 for i in range(size)] for j in range(size)]
        for i in range(size):
            for j in range(size):
                ret_map[i][j] = torch.sum(one_map[y_range[i]:y_range[i + 1], x_range[j]:x_range[j + 1]]).item()
        maps.append(ret_map)
    cfg.global_dic['confidences'] = maps


def apply_nms(confidence_map, hmap, wmap, dotmap_pred_downscale=2, thresh=0.3):
    nms_conf_map, nms_conf_box = extract_conf_points([confidence_map[0], confidence_map[1]], [hmap[0], hmap[1]])
    nms_conf_map, nms_conf_box = extract_conf_points([confidence_map[2], nms_conf_map], [hmap[2], nms_conf_box])
    nms_conf_map, nms_conf_box = extract_conf_points([confidence_map[3], nms_conf_map], [hmap[3], nms_conf_box])

    chunk_confidence_map(confidence_map, cfg.S)

    confidence_map = nms_conf_map
    hmap = nms_conf_box
    wmap = nms_conf_box

    confidence_map = np.squeeze(confidence_map)
    hmap = np.squeeze(hmap)
    wmap = np.squeeze(wmap)

    dets_idx = np.where(confidence_map > 0)

    y, x = dets_idx[-2], dets_idx[-1]
    h, w = hmap[dets_idx], wmap[dets_idx]
    x1 = x - w / 2
    x2 = x + w / 2
    y1 = y - h / 2
    y2 = y + h / 2
    scores = confidence_map[dets_idx]

    dets = np.stack([np.array(x1), np.array(y1), np.array(x2), np.array(y2), np.array(scores)], axis=1)
    boxes = torch.from_numpy(dets[:, :4]).cuda()
    confidences = torch.from_numpy(scores).cuda()
    img_size = confidence_map.shape
    cfg.global_dic['img_size'] = img_size

    # if cfg.prepare_training_data:
    #     optimize_thresh(dets, img_size)

    keep = nms(dets, 0.2)
    boxes = boxes[keep]
    cfg.global_dic['boxes'] = restore_boxes(boxes.cpu().detach().clone().numpy())
    confidences = confidences[keep]
    cfg.prepare_training_data = True
    if cfg.prepare_training_data or cfg.use_crowd_hat:
        # Feature Compression
        compression2d(boxes, confidences, img_size)
        compression1d(boxes, confidences, img_size)
    if cfg.use_crowd_hat:
        feature2d = torch.tensor(cfg.global_dic['feature2d'], dtype=torch.float32).unsqueeze(dim=0).cuda()
        feature1d = torch.tensor(cfg.global_dic['feature1d'], dtype=torch.float32).unsqueeze(dim=0).cuda()
        count = count_model(feature2d, feature1d, mode=3)
        count = int(round(count.item()))
        nms_thresholds = nms_model(feature2d.repeat(cfg.K ** 2, 1, 1, 1), feature1d.repeat(cfg.K ** 2, 1, 1),
                                   torch.tensor(cfg.global_dic['feature2d_local']).cuda())
        nms_thresholds = torch.clamp(nms_thresholds, max=1, min=0).cpu().numpy()
        boxes = region_nms(dets, img_size, nms_thresholds)
        boxes, count = alignment_scheme(boxes, count)
        cfg.global_dic['hat_count'] = count
        cfg.global_dic['hat_boxes'] = restore_boxes(boxes)
    keep = nms(dets, thresh)

    y, x = dets_idx[-2], dets_idx[-1]
    h, w = hmap[dets_idx], wmap[dets_idx]
    x = x[keep]
    y = y[keep]
    h = h[keep]
    w = w[keep]

    scores = scores[keep]
    return x, y, h, w, scores
