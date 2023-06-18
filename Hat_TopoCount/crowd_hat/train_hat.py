import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import torch
from PIL import Image
import hat_utils as utils
import data_augmentation as da
import hat_config as cfg
from evaluation import build_model
import torchvision
from evaluation import detect, process_image
from count_decoder import train_count_decoder

train_da = da.Compose([da.ToTensor(), da.RandomCrop(), da.RandomHorizontalFlip()])
test_da = da.Compose([da.ToTensor()])


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def prepare_training_data(img_root, json_root):
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
    cfg.prepare_training_data = True
    save_root = cfg.training_data_root
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    topo = build_model()
    img_list = sorted([name for name in os.listdir(img_root)])
    # pos = img_list.index('nwpu_2503.jpeg')
    # img_list = img_list[pos:]
    print(len(img_list))
    mae = 0
    mse = 0
    cnt = 0
    for name in img_list:
        # print(name)
        try:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            img_path = os.path.join(img_root, name)
            json_path = os.path.join(json_root, name.split('.')[0] + '.json')
            with open(json_path, 'r')as json_file:
                json_file = json.load(json_file)
            points = json_file['points']
            points = torch.tensor(points, dtype=torch.float32)
            json_file['points'] = points
            img = Image.open(img_path)
            img_list = [(img, json_file)]
            # transforms = test_da if int(name.split('.')[0].split('_')[1]) > 3109 else train_da
            transforms = train_da
            img_list = transforms(img_list)
            idx = 0
            for (img, target) in img_list:
                try:
                    cfg.global_dic = {}
                    cfg.train_dic = {}
                    save_name = target['id'] + '_' + str(idx)
                    cfg.global_dic['name'] = save_name
                    gt_count = target['human_num']
                    cfg.global_dic['count'] = gt_count
                    img = torchvision.transforms.ToPILImage()(img)
                    img, h_ratio, w_ratio = process_image(img)
                    if gt_count > 0:
                        target['points'][:, 0] *= w_ratio
                        target['points'][:, 1] *= h_ratio
                    cfg.global_dic['points'] = (target['points']).tolist()
                    utils.gt_distribution(cfg.global_dic['points'], cfg.global_dic['raw_img_size'])
                    count = detect(img, topo)
                    mae += abs(gt_count - count)
                    mse += (gt_count - count) ** 2
                    cnt += 1
                    idx += 1
                    print(save_name, gt_count, count, mae / cnt, (mse / cnt) ** 0.5)
                    utils.save2disk(os.path.join(save_root, save_name + '.json'))
                except Exception as e:
                    print(e)
        except Exception as e2:
            print(e2)


if __name__ == '__main__':
    # prepare_training_data(cfg.img_root, cfg.json_root)
    train_count_decoder(5, 120, resume=0)
