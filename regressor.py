# from torch.utils.data import Dataset
# import numpy as np
import torch
import torch.nn as nn


#from utils import get_gaussian_mean, rotate_img


def get_gaussian_mean(x, axis, other_axis):
    u = torch.softmax(torch.sum(x, axis=other_axis), axis=2)
    ind = torch.arange(x.shape[axis]).cuda()
    batch = x.shape[0]
    channel = x.shape[1]
    index = ind.repeat([batch, channel, 1])
    mean_position = torch.sum(index * u, dim=2)
    return mean_position


class ConvRegressor(nn.Module):
    def __init__(self, dim, chls, h, w):
        super(ConvRegressor, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=chls * 7 * 7, out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=200),
            nn.BatchNorm1d(200),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=200, out_features=2 * dim)
        self.chls = chls
        self.h = h
        self.w = w
        self.dim = dim

    def forward(self, x):
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * self.chls)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x.view(-1, self.dim, 2)


class Regressor(nn.Module):
    def __init__(self, dim, ind, chls):
        super(Regressor, self).__init__()
        self.linear = nn.Linear(2 * chls, 2 * dim)
        self.conv = nn.Conv2d(chls, dim, 1, 0)
        self.ind = float(ind)
        self.dim = dim

    def forward(self, x):
        x = torch.relu(self.conv(x))
        h_axis = 2
        w_axis = 3
        h_mean = get_gaussian_mean(x, h_axis, w_axis) / self.ind  # .unsqueeze(-1)
        w_mean = get_gaussian_mean(x, w_axis, h_axis) / self.ind  # .unsqueeze(-1)

        gaussian = torch.cat([h_mean, w_mean], axis=1)  # Bx512

        res = torch.sigmoid(self.linear(gaussian))

        return res.view(-1, self.dim, 2), gaussian


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def __call__(self, x):
        return x.view(self.shape)


class LinearRegressor(nn.Module):
    def __init__(self, dim):
        super(LinearRegressor, self).__init__()
        self.linear1 = nn.Linear(4096, 2048)
        self.linear2 = nn.Linear(2048, 2 * dim)
        # self.reshape = Reshape((batchsize, dim, 2))
        self.dim = dim

    def forward(self, x):
        res = torch.relu(self.linear1(x))
        res = torch.relu(self.linear2(res))
        return res.view(-1, self.dim, 2)


class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
        pass

    def forward(self, x):
        return x.view(x.shape[0], -1)


class RotationPredictor(nn.Module):
    def __init__(self, pose_size):
        super(RotationPredictor, self).__init__()
        self.pred = nn.Sequential(Reshape(),
                                  nn.Linear(pose_size * pose_size, 4),
                                  nn.Sigmoid())

    def forward(self, x):
        return self.pred(x)


class RotateLayer(nn.Module):
    """
    Convert the feature map back to right direction by given label

    x: BxHxW
    """

    def __init__(self):
        super(RotateLayer, self).__init__()
        pass

    def forward(self, x, label):
        # Be careful we dont't have the channel dimension here
        res = []
        for i in range(len(label)):
            res.append(torch.rot90(x[i], label[i], [0, 1]))
        # print(label)
        return torch.stack(res)

