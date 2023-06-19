import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from tqdm import tqdm
import hat_config as cfg
import json
import torch.utils.data as Data
from torch.utils.data import Dataset
from copy import deepcopy


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


class NMSDecoder(torch.nn.Module):
    def __init__(self):
        super(NMSDecoder, self).__init__()
        self.convolution2d = torch.nn.Sequential(
            torch.nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.convolution1d = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.convolution2d_local = torch.nn.Sequential(
            torch.nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.nms_nlp = torch.nn.Sequential(
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, 1)
        )

    def forward(self, feature_2d, feature_1d, feature2d_local):
        feature_2d = self.convolution2d(feature_2d)
        feature_1d = self.convolution1d(feature_1d)
        feature2d_local = self.convolution2d_local(feature2d_local)
        feature_2d = feature_2d.flatten(start_dim=1)
        feature_1d = feature_1d.flatten(start_dim=1)
        feature2d_local = feature2d_local.flatten(start_dim=1)
        global_feature = torch.cat([feature_2d, feature_1d, feature2d_local], dim=1)
        prediction = self.nms_nlp(global_feature)
        return prediction


class Trainset(Dataset):
    def __init__(self, json_root, mode="train"):
        self.features_2d = None
        self.features_1d = None
        self.features_2d_local = None
        block = 100000
        features_2d = []
        features_1d = []
        features_2d_local = []
        nms_thresholds = []
        id_std = [i for i in range(3110, 3610, 1)]
        id_std[59] = 3098
        test_list = ['nwpu_' + str(idx) + '_0.json' for idx in id_std]
        if mode == "train":
            id_std = [i for i in range(1, 3610, 1) if id_std not in id_std]
            json_list = [name for name in os.listdir(json_root) if
                         int(name.split('.')[0].split('_')[1]) in id_std
                         and int(name.split('.')[0].split('_')[-1]) < 2]
        else:
            json_list = test_list
        for path in tqdm(json_list):
            path = os.path.join(json_root, path)
            with open(path, 'r')as data:
                data = json.load(data)
            for i in range(cfg.K * cfg.K):
                features_2d.append(data['feature2d'])
                features_1d.append(data['feature1d'])
                features_2d_local.append(data['feature2d_local'][i])
                if len(features_2d) > block:
                    if self.features_2d is None:
                        self.features_2d = torch.tensor(features_2d, dtype=torch.float32).cuda()
                    else:
                        self.features_2d = torch.cat(
                            [self.features_2d, torch.tensor(features_2d, dtype=torch.float32).cuda()], dim=0)
                    del features_2d
                    features_2d = []
                    if self.features_1d is None:
                        self.features_1d = torch.tensor(features_1d, dtype=torch.float32).cuda()
                    else:
                        self.features_1d = torch.cat(
                            [self.features_1d, torch.tensor(features_1d, dtype=torch.float32).cuda()], dim=0)
                    del features_1d
                    features_1d = []
                    if self.features_2d_local is None:
                        self.features_2d_local = torch.tensor(features_2d_local, dtype=torch.float32).cuda()
                    else:
                        self.features_2d_local = torch.cat(
                            [self.features_2d_local, torch.tensor(features_2d_local, dtype=torch.float32).cuda()],
                            dim=0)
                    del features_2d_local
                    features_2d_local = []
                nms_thresholds.append([data['nms_thresholds'][i]])
        if self.features_2d is None:
            self.features_2d = torch.tensor(features_2d, dtype=torch.float32).cuda()
        else:
            self.features_2d = torch.cat(
                [self.features_2d, torch.tensor(features_2d, dtype=torch.float32).cuda()], dim=0)
        del features_2d
        if self.features_1d is None:
            self.features_1d = torch.tensor(features_1d, dtype=torch.float32).cuda()
        else:
            self.features_1d = torch.cat(
                [self.features_1d, torch.tensor(features_1d, dtype=torch.float32).cuda()], dim=0)
        del features_1d
        if self.features_2d_local is None:
            self.features_2d_local = torch.tensor(features_2d_local, dtype=torch.float32).cuda()
        else:
            self.features_2d_local = torch.cat(
                [self.features_2d_local, torch.tensor(features_2d_local, dtype=torch.float32).cuda()], dim=0)
        del features_2d_local
        self.nms_thresholds = torch.tensor(nms_thresholds, dtype=torch.float32).cuda()
        print(len(self.nms_thresholds))

    def __getitem__(self, idx):
        return self.features_2d[idx], self.features_1d[idx], self.features_2d_local[idx], self.nms_thresholds[idx]

    def __len__(self):
        return len(self.nms_thresholds)


def train_nms_decoder(n, epochs):
    """

       :param n: 训练n次，取最好的
       :param epochs: 每次训练的epoch数

       """
    val_dataset = Trainset(cfg.training_data_root, "test")
    train_dataset = Trainset(cfg.training_data_root, "train")
    # train_dataset = val_dataset
    loss_function = L1_loss()
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    times = 0
    min_loss = 1000
    min_loss_idx = 0
    sum_min = 0
    cnt = 0
    save_model = None
    for i in range(n):
        # torch.manual_seed(times)
        model = NMSDecoder().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.99))
        mean_loss = []
        cnt_min = 1000
        record = cnt_min + 1
        for epoch in range(epochs):
            if epoch >= 80 and epoch % 5 == 0:
                if cnt_min >= record:
                    sum_min += cnt_min
                    cnt += 1
                    print('average_min:', sum_min / cnt)
                    break
                record = cnt_min
            for step, (feature_2d, feature_1d, feature2d_local, gt_nms_threshold) in enumerate(train_loader):
                prediction = model(feature_2d, feature_1d, feature2d_local)
                loss = loss_function(prediction, gt_nms_threshold)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 10 == 1:
                    print('seed:', times, 'epoch:', epoch, 'step:', step, 'min_loss:', min_loss, 'min_loss_idx:',
                          min_loss_idx,
                          'current_min:', cnt_min, 'current_loss (train)', loss.item())
            losses = []
            for step, (feature_2d, feature_1d, feature2d_local, gt_nms_threshold) in enumerate(val_loader):
                prediction = model(feature_2d, feature_1d, feature2d_local)
                # prediction = torch.sum(global_grid,dim=[1,2,3]).unsqueeze(dim=1)
                loss = loss_function(prediction, gt_nms_threshold)
                loss_val = loss.data.cpu().item()
                losses.append(loss_val)
                if step % 10 == 1:
                    print('seed:', times, 'epoch:', epoch, 'step:', step, 'min_loss:', min_loss, 'min_loss_idx:',
                          min_loss_idx,
                          'current_min:', cnt_min, 'current_loss (val):', loss.item())
            mean = np.mean(losses)
            if cnt_min > mean:
                cnt_min = mean
            if min_loss > mean:
                min_loss = mean
                min_loss_idx = times
                save_model = deepcopy(model)
                torch.save(model.state_dict(),
                           os.path.join(cfg.weights_root, 'nms_decoder_' + format(min_loss, '.4f') + '.pth'))
            mean_loss.append(mean)
            print('seed:', times, 'epoch:', epoch, 'min_loss:', min_loss, 'min_loss_idx:', min_loss_idx,
                  'current_min:', cnt_min, 'current_loss:', mean)
            # scheduler.step(mean)
        times += 1
    print('average_min:', sum_min / cnt, '\n')
    torch.save(save_model.state_dict(),
               os.path.join(cfg.weights_root, 'nms_decoder_' + format(min_loss, '.4f') + '.pth'))


if __name__ == '__main__':
    train_nms_decoder(n=10, epochs=100)
