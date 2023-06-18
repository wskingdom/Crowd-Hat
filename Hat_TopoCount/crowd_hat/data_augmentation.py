import random
from torchvision.transforms import functional as F
import torch
import copy
import torchvision.transforms as ts
import numpy as np
from PIL import ImageEnhance


def augmentation_crop_count(count):
    crop_count = 6 if count == 0 else 2 if 0 < count < 100 else 4 if 100 <= count < 200 else 8 if 200 <= count < 500 \
        else 12 if 500 <= count < 1000 else 16 if 1000 <= count < 2500 else 24 if 2500 <= count < 5000 else 32
    return crop_count


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img_list):
        for t in self.transforms:
            img_list = t(img_list)
        return img_list


class ToTensor(object):

    def __call__(self, img_list):
        image = F.to_tensor(img_list[0][0])
        img_list = [(image, img_list[0][1])]
        return img_list


class RandomCrop(object):
    def __init__(self, min_w_ratio=0.4, min_h_ratio=0.4, max_w_ratio=0.6, max_h_ratio=0.6):
        self.min_w_ratio = min_w_ratio
        self.min_h_ratio = min_h_ratio
        self.max_w_ratio = max_w_ratio
        self.max_h_ratio = max_h_ratio

    def __call__(self, img_list):
        img, target = img_list[0]
        height, width = img.shape[-2:]
        count = target['human_num']
        divide_count = augmentation_crop_count(count)
        for i in range(divide_count):
            height_ratio = np.random.randint(int(self.min_h_ratio * 100), int(self.max_h_ratio * 100)) / 100
            width_ratio = np.random.randint(int(self.min_w_ratio * 100), int(self.max_w_ratio * 100)) / 100
            width_ratio = min(width_ratio, height_ratio * 1.8)
            cnt_height = int(height * height_ratio)
            cnt_width = int(width * width_ratio)
            # select upper_left_corner as original point
            right_limit = width - cnt_width
            down_limit = height - cnt_height
            xmin = np.random.randint(0, right_limit)
            ymin = np.random.randint(0, down_limit)
            new_img = F.crop(img, ymin, xmin, cnt_height, cnt_width)
            if count > 0:
                points = target['points']
                idx = torch.where(
                    (points[:, 0] >= xmin) & (points[:, 0] <= xmin + cnt_width) & (points[:, 1] >= ymin) & (
                            points[:, 1] <= ymin + cnt_height))
                points = points[idx]
                points[:, 0] = points[:, 0] - xmin
                points[:, 1] = points[:, 1] - ymin
                new_target = copy.deepcopy(target)
                new_target['points'] = points
                new_target['human_num'] = len(points)
                new_target['hw'] = [cnt_height, cnt_width]
                img_list.append([new_img, new_target])
            else:
                new_target = copy.deepcopy(target)
                new_target['hw'] = [cnt_height, cnt_width]
                new_target['human_num'] = 0
                new_target['points'] = torch.tensor([])
                img_list.append([new_img, new_target])
        return img_list


class RandomHorizontalFlip(object):
    def __call__(self, img_list):
        new_img_list = [img_list[0]]
        idx = 0
        for (image, target) in img_list:
            height, width = image.shape[-2:]
            if idx == 0:
                image = image.flip(-1)
                new_target = copy.deepcopy(target)
                if new_target['human_num'] > 0:
                    points = new_target['points']
                    points[:, 0] = width - points[:, 0]
                    new_target['points'] = points
                new_img_list.append((image, new_target))
            else:
                points = target['points']
                if random.random() > 0.5:
                    image = image.flip(-1)
                    if target['human_num'] > 0:
                        points[:, 0] = width - points[:, 0]
                target['points'] = points
                new_img_list.append((image, target))
            idx += 1
        return new_img_list


class GrayConvert(object):
    def __call__(self, img_list):
        new_img_list = copy.deepcopy(img_list)
        for (image, target) in img_list:
            image = ts.ToPILImage()(image)
            image = ts.Grayscale(1)(image)
            image = F.to_tensor(image)
            new_img_list.append((image, target))
        return new_img_list


class ColorEnhance(object):
    def __call__(self, img_list, random_line=0.5):
        new_img_list = copy.deepcopy(img_list)
        for (image, target) in img_list:
            times = 1
            image = ts.ToPILImage()(image)
            changed = False
            for i in range(times):
                new_img = copy.deepcopy(image)
                if random.random() >= random_line:
                    random_factor = np.random.randint(6.7, 15) / 10.
                    new_img = ImageEnhance.Color(new_img).enhance(random_factor)
                    changed = True
                if random.random() >= random_line:
                    random_factor = np.random.randint(6.7, 15) / 10.
                    new_img = ImageEnhance.Brightness(new_img).enhance(random_factor)
                    changed = True
                if random.random() >= random_line:
                    random_factor = np.random.randint(6.7, 15) / 10.
                    new_img = ImageEnhance.Contrast(new_img).enhance(random_factor)
                    changed = True
                if random.random() >= random_line:
                    random_factor = np.random.randint(6.7, 15) / 10.
                    new_img = ImageEnhance.Sharpness(new_img).enhance(random_factor)
                    changed = True
                if changed:
                    new_img = F.to_tensor(new_img)
                    new_img_list.append((new_img, target))
        return new_img_list
