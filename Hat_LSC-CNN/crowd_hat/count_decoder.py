import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import hat_config as cfg
import torch
from torch.utils.data import Dataset
import torch.utils.data as Data
from tqdm import tqdm
from hat_utils import L1_loss
import numpy as np
from copy import deepcopy


class CountDecoder(torch.nn.Module):
    def __init__(self):
        super(CountDecoder, self).__init__()
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

        self.conv2d = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv1d = torch.nn.Conv1d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.count_mlp = torch.nn.Sequential(
            torch.nn.Linear(384, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 32),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(32, 1)
        )

    def forward(self, feature_2d=None, feature_1d=None, mode=None):
        assert mode in [1, 2, 3]
        # mode = 2 means training 2d convolution layers, 1 means 1d, and 3 means both 2d and 1d
        prediction = None
        if mode == 1:
            feature_1d = self.convolution1d(feature_1d)
            feature_1d = self.conv1d(feature_1d)
            prediction = feature_1d.flatten(start_dim=1)

        elif mode == 2:
            feature_2d = self.convolution2d(feature_2d)
            feature_2d = self.conv2d(feature_2d)
            prediction = feature_2d.flatten(start_dim=1)

        elif mode == 3:
            feature_2d = self.convolution2d(feature_2d)
            feature_1d = self.convolution1d(feature_1d)
            feature_2d = feature_2d.flatten(start_dim=1)
            feature_1d = feature_1d.flatten(start_dim=1)
            global_feature = torch.cat([feature_2d, feature_1d], dim=1)
            prediction = self.count_mlp(global_feature)
        return prediction


class Trainset(Dataset):
    def __init__(self, json_root, mode="train"):
        self.features_2d = None
        self.features_1d = None
        counts = []
        id_std = [i for i in range(3110, 3610, 1)]
        id_std[59] = 3098
        test_list = ['nwpu_' + str(idx) + '_0.json' for idx in id_std]
        # test_list = [name for name in os.listdir(json_root) if 'nwpu' in name
        #              and int(name.split('.')[0].split('_')[1]) >= 3110
        #              and name.split('.')[0].split('_')[-1] == '0']
        if mode == "train":
            id_std = [i for i in range(1, 3610, 1) if i not in id_std]
            json_list = [name for name in os.listdir(json_root) if int(name.split('.')[0].split('_')[1]) in id_std]
                         # and name.split('.')[0].split('_')[-1] == '0']
            json_list2 = [name for name in os.listdir(json_root) if 'nwpu' not in name]
                         #and int(name.split('.')[0].split('_')[1]) < 3110]
            json_list = json_list + json_list2
        else:
            json_list = test_list
        for path in tqdm(json_list):
            path = os.path.join(json_root, path)
            with open(path, 'r')as data:
                data = json.load(data)
            if self.features_2d is None:
                self.features_2d = torch.tensor([data['feature2d']], dtype=torch.float32).cuda()
            else:
                self.features_2d = torch.cat(
                    [self.features_2d, torch.tensor([data['feature2d']], dtype=torch.float32).cuda()], dim=0)
            if self.features_1d is None:
                self.features_1d = torch.tensor([data['feature1d']], dtype=torch.float32).cuda()
            else:
                self.features_1d = torch.cat(
                    [self.features_1d, torch.tensor([data['feature1d']], dtype=torch.float32).cuda()], dim=0)
            counts.append([data['count']])
        self.counts = torch.tensor(counts, dtype=torch.float32).cuda()
        print(len(self.counts))

    def __getitem__(self, idx):
        return self.features_2d[idx], self.features_1d[idx], self.counts[idx]

    def __len__(self):
        return len(self.counts)


def train_count_decoder(n, epochs, resume=0):
    assert resume in [0, 1, 2]
    if not os.path.exists(cfg.weights_root):
        os.mkdir(cfg.weights_root)
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
        batch_size=10000,
        shuffle=True,
        num_workers=0
    )
    save_model1 = None
    save_model2 = None
    save_model3 = None
    mode_idx = resume
    modes = [1, 2, 3]
    for mode in modes[mode_idx:]:
        times = 0
        min_loss = 1000
        min_loss_idx = 0
        sum_min = 0
        cnt = 0
        if resume > 0:
            load_models = sorted(
                [name for name in os.listdir(cfg.weights_root) if int(name.split('_')[2]) == mode_idx - 1])
            checkpoint = CountDecoder().cuda()
            print('load from ' + load_models[0])
            checkpoint.load_state_dict(torch.load(os.path.join(cfg.weights_root, load_models[0])))
            if mode_idx == 1:
                save_model1 = checkpoint
            elif mode_idx == 2:
                save_model2 = checkpoint
        for i in range(n):
            # torch.manual_seed(times)
            model = None
            if mode_idx == 0:
                model = CountDecoder().cuda()

            elif mode_idx == 1:
                model = deepcopy(save_model1)

            elif mode_idx == 2:
                model = deepcopy(save_model2)
                for param in model.convolution2d.parameters():
                    param.requires_grad = False
                for param in model.convolution1d.parameters():
                    param.requires_grad = False
                for param in model.conv2d.parameters():
                    param.requires_grad = False
                for param in model.conv1d.parameters():
                    param.requires_grad = False

            params = [p for p in model.parameters() if p.requires_grad]
            learning_rate = 0.0003 if mode_idx < 2 else 0.0001
            optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.9, 0.99))
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
                for step, (feature_2d, feature_1d, count) in enumerate(train_loader):
                    prediction = model(feature_2d, feature_1d, mode)
                    loss = loss_function(prediction, count)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                losses = []
                for step, (feature_2d, feature_1d, count) in enumerate(val_loader):
                    prediction = model(feature_2d, feature_1d, mode)
                    loss = loss_function(prediction, count)
                    loss_val = loss.data.cpu().item()
                    losses.append(loss_val)
                mean = np.mean(losses)
                if cnt_min > mean:
                    cnt_min = mean
                if min_loss > mean:
                    min_loss = mean
                    min_loss_idx = times
                    if mode_idx == 0:
                        save_model1 = deepcopy(model)
                    elif mode_idx == 1:
                        save_model2 = deepcopy(model)
                    elif mode_idx == 2:
                        save_model3 = deepcopy(model)
                    torch.save(model.state_dict(), os.path.join(cfg.weights_root,
                                                                'count_decoder_%s_' % mode_idx + format(min_loss,
                                                                                                        '.4f') + '.pth'))
                mean_loss.append(mean)
                print('seed:', times, 'epoch:', epoch, 'min_loss:', min_loss, 'min_loss_idx:', min_loss_idx,
                      'current_min:', cnt_min, 'current_loss:', mean)
                # scheduler.step(mean)

            times += 1
        print('average_min:', sum_min / cnt, '\n')
        mode_idx += 1
    torch.save(save_model3.state_dict(),
               os.path.join(cfg.weights_root, 'count_decoder_%s_' % mode_idx + format(min_loss, '.4f') + '.pth'))


if __name__ == '__main__':
    train_count_decoder(20, 120, resume=0)
