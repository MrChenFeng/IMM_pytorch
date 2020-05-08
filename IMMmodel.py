import torch.nn as nn
from utils import get_gaussian_mean
import torch


class IMM(nn.Module):
    def __init__(self, dim=10):
        """
        It should be noted all params has been fixed to Jakab 2018 paper.
        Goto the original class if params and layers need to be changed.
        Images should be rescaled to 128*128
        """
        super(IMM, self).__init__()
        self.content_encoder = Encoder()
        self.pose_encoder = PoseEncoder(dim)
        self.generator = Generator()

    def forward(self, x, y):
        content_x = self.content_encoder(x)
        pose_y, pose_coord = self.pose_encoder(y)
        code = torch.cat((content_x, pose_y), dim=1)
        recovered_y = self.generator(code)
        return recovered_y, pose_coord


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1_1 = self._gen_conv_block(3, 32, (7, 7), 1, (3, 3))
        self.conv1_2 = self._gen_conv_block(32, 32, (3, 3), 1, (1, 1))

        self.conv2_1 = self._gen_conv_block(32, 64, (3, 3), 2, (1, 1))
        self.conv2_2 = self._gen_conv_block(64, 64, (3, 3), 1, (1, 1))

        self.conv3_1 = self._gen_conv_block(64, 128, (3, 3), 2, (1, 1))
        self.conv3_2 = self._gen_conv_block(128, 128, (3, 3), 1, (1, 1))

        self.conv4_1 = self._gen_conv_block(128, 256, (3, 3), 2, (1, 1))
        self.conv4_2 = self._gen_conv_block(256, 256, (3, 3), 1, (1, 1))

        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.conv4_2
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    @staticmethod
    def _gen_conv_block(inc, outc, size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=size, stride=stride, padding=padding),
            nn.BatchNorm2d(outc),
            nn.ReLU()
        )


class PoseEncoder(Encoder):
    def __init__(self, dim=10):
        """

        Args:
            dim (int): Num of keypoints
        """
        super(PoseEncoder, self).__init__()
        # It should be noted dim1 and dim2 should be consistent with Encoder
        dim1 = 256
        dim2 = 16
        self.final_conv = nn.Conv2d(dim1, dim, (1, 1), 1, (0, 0))
        self.heatmap = HeatMap(0.1, (dim2, dim2))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.final_conv(x)
        heatmap, coord = self.heatmap(x)
        return heatmap, coord


class Generator(nn.Module):
    """"""

    def __init__(self, map_size=[16, 16], channels=256 + 10):
        super(Generator, self).__init__()
        self.conv1_1 = self._gen_conv_block(channels, 128, (3, 3), 1, (1, 1))
        self.conv1_2 = self._gen_conv_block(128, 128, (3, 3), 1, (1, 1))

        map_size = [2 * s for s in map_size]
        self.upsample1 = nn.Upsample(map_size)
        self.conv2_1 = self._gen_conv_block(128, 64, (3, 3), 1, (1, 1))
        self.conv2_2 = self._gen_conv_block(64, 64, (3, 3), 1, (1, 1))

        map_size = [2 * s for s in map_size]
        self.upsample2 = nn.Upsample(map_size)
        self.conv3_1 = self._gen_conv_block(64, 32, (3, 3), 1, (1, 1))
        self.conv3_2 = self._gen_conv_block(32, 32, (3, 3), 1, (1, 1))

        map_size = [2 * s for s in map_size]
        self.upsample3 = nn.Upsample(map_size)
        self.conv4_1 = self._gen_conv_block(32, 32, (3, 3), 1, (1, 1))
        self.conv4_2 = self._gen_conv_block(32, 32, (3, 3), 1, (1, 1))

        self.final_conv = nn.Conv2d(32, 3, (3, 3), 1, (1, 1))
        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.upsample1,
            self.conv2_1,
            self.conv2_2,
            self.upsample2,
            self.conv3_1,
            self.conv3_2,
            self.upsample3,
            self.conv4_1,
            self.conv4_2,
            self.final_conv
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    @staticmethod
    def _gen_conv_block(inc, outc, size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=size, stride=stride, padding=padding),
            nn.BatchNorm2d(outc),
            nn.ReLU()
        )


class HeatMap(nn.Module):
    """
    Refine the estimated pose map to be gaussian distributed heatmap.
    Calculate the gaussian mean value.
    Params:
    std: standard deviation of gaussian distribution
    output_size: output feature map size
    """

    def __init__(self, std, output_size):
        super(HeatMap, self).__init__()
        self.std = std
        self.out_h, self.out_w = output_size

    def forward(self, x, h_axis=2, w_axis=3):
        """
        x: feature map BxCxHxW
        h_axis: the axis of Height
        w_axis: the axis of width
        """
        # self.in_h, self.in_w = x.shape[h_axis:]
        batch, channel = x.shape[:h_axis]

        # Calculate weighted position of joint(-1~1)
        # h_scale = self.in_h / float(self.out_h)
        # w_scale = self.in_w / float(self.out_w)
        h_mean = get_gaussian_mean(x, h_axis, w_axis)  # BxC
        w_mean = get_gaussian_mean(x, w_axis, h_axis)  # BxC

        coord = torch.stack([h_mean, w_mean], dim=2)

        h_mean = h_mean.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.out_h, self.out_w)
        w_mean = w_mean.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.out_h, self.out_w)

        h_ind = torch.tensor(torch.linspace(-1, 1, self.out_h)).unsqueeze(-1).repeat(batch, channel, 1, self.out_w).to(
            x.device)
        w_ind = torch.tensor(torch.linspace(-1, 1, self.out_w)).unsqueeze(0).repeat(batch, channel, self.out_h, 1).to(
            x.device)
        dist = (h_ind - h_mean) ** 2 + (w_ind - w_mean) ** 2

        # div = dist.sum(dim=[2, 3], keepdim=True).repeat(1, 1, self.out_h, self.out_w)
        # dist = dist * self.out_w * self.out_h / div

        # torch.normal()
        res = torch.exp(-dist / self.std ** 2)
        return res , coord


if __name__ == '__main__':
    t = IMM()
    x = torch.randn(10, 3, 128, 128)
    # x: the original image
    y = torch.randn(10, 3, 128, 128)
    # y: the warped image
    from torchsummary import summary

    summary(t, (x, y))
