import torch
import hat_config as cfg
import numpy as np
from f1_evaluation import hungarian
from scipy import spatial as ss
from PIL import ImageDraw
import cv2
import json
import os


class L1_loss(torch.nn.Module):

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


def load_crowd_hat(model_path):
    try:
        from count_decoder import CountDecoder
        count_model = CountDecoder()
        count_model.load_state_dict(
            torch.load(model_path, map_location='cpu'))
        count_model.cuda().eval()
    except Exception as e:
        print(e)
        count_model = None
        cfg.use_crowd_hat = False
    cfg.global_dic['count_model'] = count_model


def get_f1(pred_p, gt_p, sigma):
    """
    compute the F1-measure proposed in "NWPU-Crowd: A Large-Scale Benchmark for Crowd Counting and Localization"
    @param pred_p:  prediction point set
    @param gt_p:    ground truth point set
    @param sigma:   match distance
    @return:
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
    pred_num = pred_p.shape[0]
    gt_num = gt_p.shape[0]
    level = np.ones(gt_num)
    for i_pred_p in range(pred_num):
        pred_dist = dist_matrix[i_pred_p, :]
        match_matrix[i_pred_p, :] = pred_dist <= sigma

    tp, assign = hungarian(match_matrix)
    fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
    tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
    tp_gt_index = np.array(np.where(assign.sum(0) == 1))[0]
    fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

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
    :param boxes: boxes after region-adaptive NMS
    :param count: the predicted count from count decoder
    """
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
    get the distribution of ground truth point set. split the set into K * K patch
    :param points: ground truth point set
    :param hw: size of input image
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


def resize_image(image):
    """
    opencv image format
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
    split global information into K*K local information corresponding with K*K regions in the image
    :param global_info: 2D compressed map
    """
    size = cfg.S
    divide_size = cfg.K
    # tensor变换，功能是快速拆分一个map为多个等边长的子map
    a = global_info.detach().clone()
    channels = a.shape[0]
    b = torch.chunk(a, divide_size, 2)
    b = torch.stack(b, dim=1)
    b = torch.chunk(b, divide_size, 2)
    b = torch.stack(b, dim=1)
    b = b.reshape(channels, divide_size * divide_size, int(size / divide_size), int(size / divide_size))
    b = b.permute(1, 0, 2, 3)
    cfg.global_dic['feature2d_local'] = b.cpu().tolist()


def draw_box(image, hat_boxes, detection_boxes, thickness):
    """
    for visualization
    """
    hat_boxes = (hat_boxes[:, :4]).astype('int').tolist()
    hat_boxes = tuple(map(tuple, hat_boxes))
    detection_boxes = (detection_boxes[:, :4]).astype('int').tolist()
    detection_boxes = tuple(map(tuple, detection_boxes))
    intersection = list(set(hat_boxes) & set(detection_boxes))
    hat_only = list(set(hat_boxes) - set(detection_boxes))
    detection_only = list(set(detection_boxes) - set(hat_boxes))
    for box in intersection:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), thickness)
    for box in hat_only:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), thickness)
    for box in detection_only:
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), thickness)
    cv2.imwrite('vis/result/%s.png' % (cfg.global_dic['name']), image)


# def find_thresh_by_f1(dets, gt_p):
#     gt_p = np.array(gt_p)
#     best_f1 = -1
#     best_thresh = -1
#     record = -1
#     for i in range(10, 30):
#         score_thresh = i / 100
#         # keep = (dets[:, 4] >= score_thresh)
#         keep = nms(dets, score_thresh)
#         new_dets = dets[keep]
#         h_ratio = cfg.global_dic['raw_img_size'][0] / cfg.global_dic['img_size'][0]
#         w_ratio = cfg.global_dic['raw_img_size'][1] / cfg.global_dic['img_size'][1]
#         x_ctr = (new_dets[:, 0] + new_dets[:, 2]) * w_ratio / 2
#         y_ctr = (new_dets[:, 1] + new_dets[:, 3]) * h_ratio / 2
#         pred_p = np.stack((x_ctr, y_ctr), axis=1)
#         f1 = get_f1(pred_p, gt_p, 15)
#         if score_thresh == cfg.default_thresh:
#             record = f1
#         if f1 > best_f1:
#             best_f1 = f1
#             best_thresh = score_thresh
#     # print('best_f1:', best_f1, 'default_f1:', record, 'best_threshold:', best_thresh, 'improve:', best_f1 - record)
#     assert best_thresh >= 0
#     return best_thresh
#
#
# def optimize_thresh(dets, image_shape):
#     divide_size = cfg.K
#     dets = torch.from_numpy(dets).cuda()
#     x_unit = image_shape[1] / divide_size
#     y_unit = image_shape[0] / divide_size
#     x_ctr = (dets[:, 0] + dets[:, 2]) / 2
#     y_ctr = (dets[:, 1] + dets[:, 3]) / 2
#     x_pos = torch.clamp(torch.floor(x_ctr / x_unit), max=divide_size - 1)
#     y_pos = torch.clamp(torch.floor(y_ctr / y_unit), max=divide_size - 1)
#     pos = y_pos * divide_size + x_pos
#     block_num = divide_size * divide_size
#     local_threshes = [0 for i in range(block_num)]
#     f1s = []
#     raws = []
#     for i in range(block_num):
#         idxes = (pos == i)
#         boxes = dets[idxes]
#         local_threshes[i] = find_thresh_by_f1(boxes.cpu().numpy(), cfg.global_dic['point_dist'][i])
#     cfg.global_dic['nms_thresholds'] = local_threshes


def feature_visualization():
    # names = ['area2d', 'confidence2d', 'count2d']
    names = ['area2d', 'confidence2d']
    size = cfg.S
    new_size = cfg.S * 20  # *20 to magnify the resolution
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
        vis_img = np.repeat(data.T, 160, 1)  # repeat 32 times to magnify the resolution
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-7)
        vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
        cfg.global_dic[name] = vis_img


def compression2d(x_ctr, y_ctr, areas, confidences, image_shape):
    # image_shape : [h, w]
    size = cfg.S
    x_unit = image_shape[1] / size
    y_unit = image_shape[0] / size
    areas = areas * cfg.alpha_ba / (image_shape[0] * image_shape[1])
    areas = areas.cpu().tolist()
    x_pos = torch.clamp(torch.floor(x_ctr / x_unit), max=size - 1)
    y_pos = torch.clamp(torch.floor(y_ctr / y_unit), max=size - 1)
    pos = y_pos * size + x_pos
    pos = pos.cpu().numpy().astype('int32').tolist()
    box_count = len(x_ctr)
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
    topological_map = torch.tensor(cfg.global_dic['topological'])
    feature2d = torch.stack([area2d, confidence2d, count2d], dim=0)
    feature2d = torch.cat((topological_map, feature2d), dim=0)
    local_info(feature2d)
    cfg.global_dic['feature2d'] = feature2d.cpu().tolist()


def compression1d(areas, confidences, image_shape):
    length = cfg.L
    total_area = image_shape[0] * image_shape[1]
    areas = areas * cfg.alpha_ba / total_area
    areas = torch.tanh(areas)
    vectors = [areas, confidences]  # vectors of area sizes and confidence scores before feature compression
    compressed_vectors = []
    box_count = len(areas)
    for vector in vectors:
        pos = torch.clamp(torch.floor(vector * length), min=0, max=length - 1)
        pos = pos.cpu().numpy().astype('int32').tolist()
        # Using List to process is faster than ndarray or tensor
        compressed_vector = [0 for i in range(length)]
        for i in range(box_count):
            compressed_vector[pos[i]] += 1
        compressed_vectors.append(compressed_vector)
    area1d = torch.tensor(compressed_vectors[0])
    confidence1d = torch.tensor(compressed_vectors[1])
    cfg.global_dic['feature1d'] = torch.stack([area1d, confidence1d], dim=0).cpu().tolist()


def compression(one_map):
    if type(one_map) is np.ndarray:
        one_map = torch.from_numpy(one_map)
    size = cfg.S
    h, w = one_map.shape
    x_unit = w / size
    y_unit = h / size
    x_range = np.ceil(np.arange(size + 1) * x_unit).astype('int32').tolist()
    y_range = np.ceil(np.arange(size + 1) * y_unit).astype('int32').tolist()
    ret_map = np.zeros((size, size)).tolist()
    for i in range(size):
        for j in range(size):
            ret_map[i][j] = torch.sum(one_map[y_range[i]:y_range[i + 1], x_range[j]:x_range[j + 1]]).item()
    deviation = torch.abs(torch.sum(one_map) - np.sum(ret_map)).item()
    assert deviation < 1, deviation
    return ret_map


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
