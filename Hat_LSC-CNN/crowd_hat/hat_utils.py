import torch
import hat_config as cfg
import numpy as np
from crowd_hat.f1_evaluation import hungarian
from scipy import spatial as ss
from PIL import ImageDraw
import cv2
import json
from utils.nms import nms
import os


class L1_loss(torch.nn.Module):
    """
    论文中采用的损失函数
    """

    def __init__(self):
        super(L1_loss, self).__init__()

    def forward(self, prediction, gt):
        loss = torch.abs(prediction - gt)
        loss = torch.mean(loss)
        return loss


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, prediction, gt):
        loss = torch.pow((prediction - gt), 2)
        loss = torch.mean(loss)
        loss = torch.sqrt(loss)
        return loss


class ThreshLoss(torch.nn.Module):
    def __init__(self):
        super(ThreshLoss, self).__init__()

    def forward(self, prediction, gt_thresh, gt_count):
        loss = torch.abs(prediction - gt_thresh)
        loss *= torch.sqrt(gt_count)
        loss = torch.mean(loss)
        return loss


def evaluate_f1_with_box(pred_p, gt_p, sigma):
    """
    根据预测点集和真实点集，以及约束参数sigma(见NWPU-Crowd)，计算F1-Score
    :param pred_p: 预测点集
    :param gt_p: 真实点集
    :param sigma: 约束参数
    """
    gt_num = gt_p.shape[0]
    pred_num = pred_p.shape[0]
    if gt_num == 0 and pred_num == 0:
        return 0, 0, 0
    if gt_num == 0 and pred_num != 0:
        return 0, pred_num, 0
    if gt_num != 0 and pred_num == 0:
        return 0, 0, gt_num
    num_classes = 6
    dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
    match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
    level = np.ones(gt_num)
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= sigma
    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]
    level_list = level[tp_gt_index]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    tp_c = np.zeros([num_classes])
    fn_c = np.zeros([num_classes])

    for i_class in range(num_classes):
        tp_c[i_class] = (level[tp_gt_index] == i_class).sum()
        fn_c[i_class] = (level[fn_gt_index] == i_class).sum()

    return tp, fp, fn


def get_f1(pred_p, gt_p, sigma=15):
    """
    根据预测点集和真实点集，以及约束参数sigma(见NWPU-Crowd)，计算F1-Score
    :param pred_p: 预测点集
    :param gt_p: 真实点集
    :param sigma: 约束参数
    """
    gt_num = gt_p.shape[0]
    pred_num = pred_p.shape[0]
    if gt_num == 0 and pred_num == 0:
        return 1
    if gt_num == 0 or pred_num == 0:
        return 0
    num_classes = 6
    dist_matrix = ss.distance_matrix(pred_p, gt_p, p=2)
    match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
    level = np.ones(gt_num)
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= sigma

    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]
    level_list = level[tp_gt_index]

    tp = tp_pred_index.shape[0]
    fp = fp_pred_index.shape[0]
    fn = fn_gt_index.shape[0]

    tp_c = np.zeros([num_classes])
    fn_c = np.zeros([num_classes])

    for i_class in range(num_classes):
        tp_c[i_class] = (level[tp_gt_index] == i_class).sum()
        fn_c[i_class] = (level[fn_gt_index] == i_class).sum()

    if tp == 0:
        return 0
    precision = tp / (tp + fp + 1e-20)
    recall = tp / (tp + fn + 1e-20)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def region_nms(dets, image_shape, nms_thresholds):
    divide_size = cfg.K
    dets = torch.from_numpy(dets).cuda()
    x_unit = image_shape[1] / divide_size
    y_unit = image_shape[0] / divide_size
    x_ctr = (dets[:, 0] + dets[:, 2]) / 2
    y_ctr = (dets[:, 1] + dets[:, 3]) / 2
    x_pos = torch.clamp(torch.floor(x_ctr / x_unit), max=divide_size - 1)
    y_pos = torch.clamp(torch.floor(y_ctr / y_unit), max=divide_size - 1)
    pos = y_pos * divide_size + x_pos
    block_num = divide_size * divide_size
    # block_num = 16
    all_boxes = []
    for i in range(block_num):
        idxes = (pos == i)
        boxes = dets[idxes].cpu().numpy()
        if boxes.shape[0] == 0:
            continue
        region_nms_threshold = nms_thresholds[i]
        keep = nms(boxes, region_nms_threshold)
        boxes = boxes[keep]
        all_boxes.append(boxes)
    if all_boxes:
        all_boxes = np.concatenate(all_boxes, axis=0)
        return all_boxes
    else:
        return np.empty((0, 4))


def draw_point(image, points, r=2):
    # Draw all boxes onto image.
    draw = ImageDraw.Draw(image)
    for point in points:
        x = point[0]
        y = point[1]
        leftUpPoint = (x - r, y - r)
        rightDownPoint = (x + r, y + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, fill='Chartreuse')


def alignment_scheme(boxes, count):
    """
    :param boxes: 经过nms decoder得到的boxes
    :param count: count decoder预测人数
    """
    if boxes.any():
        scores = boxes[:, 4]
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        box_count = len(boxes)
        if box_count > count:
            boxes = boxes[:count]
        else:
            count = box_count
        assert len(boxes) == count
    return boxes, count


def gt_distribution(points, hw):
    """
    训练时寻找最合适的区域f1，需要得到每一块区域的gt标注
    :param points: 整张图的gt标注
    :param hw: 图片尺寸，用来计算每一块区域的宽(x_unit)和高(y_unit)
    """
    size = cfg.K
    if not points:
        cfg.global_dic['point_dist'] = [[] for i in range(size * size)]
        return
    h, w = hw
    x_unit = w / size
    y_unit = h / size
    points = torch.tensor(points).cuda()
    x_ctr = points[:, 0]
    y_ctr = points[:, 1]
    x_pos = torch.clamp(torch.floor(x_ctr / x_unit), max=size - 1, min=0)
    y_pos = torch.clamp(torch.floor(y_ctr / y_unit), max=size - 1, min=0)
    pos = y_pos * size + x_pos
    point_dist = []
    for i in range(size * size):
        idx_i = (pos == i)
        points_i = points[idx_i].tolist()
        point_dist.append(points_i)
    cfg.global_dic['point_dist'] = point_dist


def model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')


class ConfidenceLoss(torch.nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def forward(self, prediction, gt_con, gt_weight):
        loss = torch.nn.BCEWithLogitsLoss()(prediction, gt_con)
        return loss


def get_count(scores, pos):
    return torch.where(torch.gt(scores, pos))[0].shape[0]


def get_global_dic():
    return cfg.global_dic


def resize_image(image):
    """
    image为cv2格式
    """
    max_size = 2048
    min_size = 512
    h, w = image.shape[:2]
    if h < min_size or w < min_size:
        ratio = max(min_size / h, min_size / w)
    elif h <= min_size * 2 and w <= min_size * 2:
        ratio = 1
    else:
        ratio = min(max_size / h, max_size / w, 0.5)
    if ratio != 1:
        w, h = int(w * ratio), int(h * ratio)
        image = cv2.resize(image, (w, h))
    cfg.global_dic['raw_img_size'] = [h, w]
    return image, ratio


def local_info(global_info):
    """
    将一个 2d compressed map拆分成 K*K块 local map
    :param global_info: 2d compressed map
    """
    size = cfg.S
    divide_size = cfg.K
    # tensor变换，功能是快速拆分一个map为多个等边长的子map
    a = global_info[0]  # 3 * 64 * 64
    b = torch.chunk(a, divide_size, 2)
    b = torch.stack(b, dim=1)
    b = torch.chunk(b, divide_size, 2)
    b = torch.stack(b, dim=1)
    b = b.reshape(7, 16, int(size / divide_size), int(size / divide_size))
    b = b.permute(1, 0, 2, 3)
    cfg.global_dic['feature2d_local'] = b.cpu().tolist()


def restore_boxes(boxes):
    if boxes.any():
        h_ratio = cfg.global_dic['raw_img_size'][0] / cfg.global_dic['img_size'][0]
        w_ratio = cfg.global_dic['raw_img_size'][1] / cfg.global_dic['img_size'][1]
        boxes[:, (0, 2)] *= w_ratio
        boxes[:, (1, 3)] *= h_ratio
    return boxes


def draw_box(image, hat_boxes, detection_boxes, thickness):
    """
    检测结果可视化
    """
    hat_boxes = (hat_boxes[:, :4]).astype('int').tolist()
    hat_boxes = tuple(map(tuple, hat_boxes))
    detection_boxes = (detection_boxes[:, :4]).astype('int').tolist()
    detection_boxes = tuple(map(tuple, detection_boxes))
    intersection = list(set(hat_boxes) & set(detection_boxes))
    hat_only = list(set(hat_boxes) - set(detection_boxes))
    detection_only = list(set(detection_boxes) - set(hat_boxes))
    for box in intersection:  # 绿色框为原来检测的以及使用了nms decoder后共有的框
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), thickness)
    for box in hat_only:  # 只在使用了nms decoder才有的框
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), thickness)
    for box in detection_only:  # 只在原来检测框中存在的框
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness)
    cv2.imwrite('vis/result/%s.png' % (cfg.global_dic['name']), image)


def find_thresh_by_f1(dets, gt_p):
    gt_p = np.array(gt_p)
    best_f1 = -1
    best_thresh = -1
    record = -1
    for i in range(10, 30):
        score_thresh = i / 100
        # keep = (dets[:, 4] >= score_thresh)
        keep = nms(dets, score_thresh)
        new_dets = dets[keep]
        h_ratio = cfg.global_dic['raw_img_size'][0] / cfg.global_dic['img_size'][0]
        w_ratio = cfg.global_dic['raw_img_size'][1] / cfg.global_dic['img_size'][1]
        x_ctr = (new_dets[:, 0] + new_dets[:, 2]) * w_ratio / 2
        y_ctr = (new_dets[:, 1] + new_dets[:, 3]) * h_ratio / 2
        pred_p = np.stack((x_ctr, y_ctr), axis=1)
        f1 = get_f1(pred_p, gt_p, 15)
        if score_thresh == cfg.default_thresh:
            record = f1
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = score_thresh
    # print('best_f1:', best_f1, 'default_f1:', record, 'best_threshold:', best_thresh, 'improve:', best_f1 - record)
    assert best_thresh >= 0
    return best_thresh


def optimize_thresh(dets, image_shape):
    divide_size = cfg.K
    dets = torch.from_numpy(dets).cuda()
    x_unit = image_shape[1] / divide_size
    y_unit = image_shape[0] / divide_size
    x_ctr = (dets[:, 0] + dets[:, 2]) / 2
    y_ctr = (dets[:, 1] + dets[:, 3]) / 2
    x_pos = torch.clamp(torch.floor(x_ctr / x_unit), max=divide_size - 1)
    y_pos = torch.clamp(torch.floor(y_ctr / y_unit), max=divide_size - 1)
    pos = y_pos * divide_size + x_pos
    block_num = divide_size * divide_size
    local_threshes = [0 for i in range(block_num)]
    f1s = []
    raws = []
    for i in range(block_num):
        idxes = (pos == i)
        boxes = dets[idxes]
        local_threshes[i] = find_thresh_by_f1(boxes.cpu().numpy(), cfg.global_dic['point_dist'][i])
    cfg.global_dic['nms_thresholds'] = local_threshes


def feature_visualization():
    """
    Feature Compression可视化
    """
    # names = ['area2d', 'confidence2d', 'count2d']
    names = ['area2d', 'confidence2d']
    size = cfg.S
    new_size = cfg.S * 20  # *20是为了提高分辨率，出图更加清晰
    step = new_size // size
    step_h = int(1080 / 64)
    step_w = int(1920 / 64)
    for name in names:
        data = np.array(cfg.global_dic[name])
        vis_img = np.zeros((size * step_h, size * step_w))
        for i in range(size):
            for j in range(size):
                vis_img[i * step_h:(i + 1) * step_h, j * step_w:(j + 1) * step_w] = data[i, j]
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-7)
        vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
        cfg.global_dic[name] = vis_img
        # cv2.imwrite('vis/2d/%s_%s.png' % (cfg.global_dic['name'], name), vis_img)

    names = ['area1d', 'confidence1d']
    for name in names:
        arr = np.array(cfg.global_dic[name]).repeat(4).tolist()
        data = np.array([arr])
        vis_img = np.repeat(data.T, 160, 1)  # 重复维度32次也是为了提高分辨率
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-7)
        vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
        cfg.global_dic[name] = vis_img
        # cv2.imwrite('vis/1d/%s_%s.png' % (cfg.global_dic['name'], name), vis_img)

    # names = ['area2d', 'confidence2d', 'count2d']
    # size = int(cfg.S / cfg.K)
    # new_size = size * 20
    # step = new_size // size
    # assets = np.array(cfg.global_dic['feature2d_local'])
    # for idx in range(len(names)):
    #     feature = assets[:, idx, :, :]
    #     name = names[idx]
    #     image_list = []
    #     for pos in range(cfg.K ** 2):
    #         data = feature[pos]
    #         vis_img = np.zeros((new_size, new_size))
    #         for i in range(size):
    #             for j in range(size):
    #                 vis_img[i * step:(i + 1) * step, j * step:(j + 1) * step] = data[i, j]
    #         vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-7)
    #         vis_img = (vis_img * 255).astype(np.uint8)
    #         vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_TURBO)
    #         image_list.append(vis_img)
    #         cv2.imwrite('vis/2d_local/%s_%s_%s.png' % (cfg.global_dic['name'], name, pos), vis_img)
    #     img = np.zeros((new_size * cfg.K, new_size * cfg.K, 3))
    #     for i in range(cfg.K):
    #         for j in range(cfg.K):
    #             img[i * new_size:(i + 1) * new_size, j * new_size:(j + 1) * new_size, :] = image_list[i * cfg.K + j]
    #     cv2.imwrite('vis/2d_local/%s_%s.png' % (cfg.global_dic['name'], name), img)


def compression2d(boxes, confidences, image_shape):
    # image_shape : [h, w]
    size = cfg.S
    x_unit = image_shape[1] / size
    y_unit = image_shape[0] / size
    x_ctr = (boxes[:, 0] + boxes[:, 2]) / 2
    y_ctr = (boxes[:, 1] + boxes[:, 3]) / 2
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (image_shape[0] * image_shape[1])
    areas = areas.cpu().tolist()
    x_pos = torch.clamp(torch.floor(x_ctr / x_unit), max=size - 1)
    y_pos = torch.clamp(torch.floor(y_ctr / y_unit), max=size - 1)
    pos = y_pos * size + x_pos
    pos = pos.cpu().numpy().astype('int32').tolist()
    box_count = len(boxes)
    conf_map = [0 for i in range(size * size)]
    area_map = [0 for i in range(size * size)]
    count_map = [0 for i in range(size * size)]
    confidences = confidences.cpu().tolist()
    for i in range(box_count):
        area_map[pos[i]] += areas[i]
        conf_map[pos[i]] += confidences[i]
        count_map[pos[i]] += 1
    confidence2d = torch.tensor(conf_map).reshape([size, size])
    area2d = torch.tensor(area_map).reshape([size, size])
    count2d = torch.tensor(count_map).reshape([size, size])
    confidences_map = torch.tensor(cfg.global_dic['confidences']).reshape([4, size, size])
    feature2d = torch.stack([area2d, confidence2d, count2d], dim=0)
    feature2d = torch.cat((confidences_map, feature2d), dim=0)
    local_info(feature2d.unsqueeze(dim=0))
    cfg.global_dic['feature2d'] = feature2d.cpu().tolist()


def save2disk(save_path):
    if not os.path.exists(cfg.training_data_root):
        os.mkdir(cfg.training_data_root)
    save_dic = {}
    # train_data = ['feature2d', 'feature1d', 'feature2d_local', 'count', 'nms_thresholds']
    train_data = ['feature2d', 'feature1d', 'count']
    for item in train_data:
        save_dic[item] = cfg.global_dic[item]
    with open(save_path, 'w')as json_file:
        json.dump(save_dic, json_file)


def compression1d(boxes, confidences, image_shape):
    """

    :param boxes: bounding boxes,用来计算面积
    :param confidences:
    :param image_shape: 用来面积归一化
    :return:
    """
    length = cfg.L
    total_area = image_shape[0] * image_shape[1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * cfg.alpha_ba / total_area
    areas = torch.tanh(areas)
    vectors = [areas, confidences]  # 未压缩的areas和confidences向量
    compressed_vectors = []
    box_count = len(boxes)
    for vector in vectors:
        pos = torch.clamp(torch.floor(vector * length), min=0, max=length - 1)
        pos = pos.cpu().numpy().astype('int32').tolist()
        # 用list比tensor和ndarray更快
        compressed_vector = [0 for i in range(length)]
        for i in range(box_count):
            compressed_vector[pos[i]] += 1
        compressed_vectors.append(compressed_vector)
    area1d = torch.tensor(compressed_vectors[0])
    confidence1d = torch.tensor(compressed_vectors[1])
    cfg.global_dic['feature1d'] = torch.stack([area1d, confidence1d], dim=0).cpu().tolist()
