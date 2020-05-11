import torch
import torch.nn as nn

from utils import get_gaussian_mean


class IMM(nn.Module):
    def __init__(self, dim=10, heatmap_std=0.1, h_channel=64):
        """
        It should be noted all params has been fixed to Jakab 2018 paper.
        Goto the original class if params and layers need to be changed.
        Images should be rescaled to 128*128
        """
        super(IMM, self).__init__()
        self.content_encoder = Encoder(h_channel)
        self.pose_encoder = PoseEncoder(dim, heatmap_std,h_channel)
        self.generator = Generator(h_channel=h_channel)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        content_x = self.content_encoder(x)
        pose_y, pose_coord = self.pose_encoder(y)
        code = torch.cat((content_x, pose_y), dim=1)
        recovered_y = self.generator(code)
        return recovered_y, pose_coord


class Encoder(nn.Module):
    def __init__(self, h_channel=64):
        super(Encoder, self).__init__()
        self.conv1_1 = self._gen_conv_block(3, h_channel, (7, 7), 1, (3, 3))
        self.conv1_2 = self._gen_conv_block(h_channel, h_channel, (3, 3), 1, (1, 1))

        self.conv2_1 = self._gen_conv_block(h_channel, 2 * h_channel, (3, 3), 2, (1, 1))
        self.conv2_2 = self._gen_conv_block(2 * h_channel, 2 * h_channel, (3, 3), 1, (1, 1))

        self.conv3_1 = self._gen_conv_block(2 * h_channel, 4 * h_channel, (3, 3), 2, (1, 1))
        self.conv3_2 = self._gen_conv_block(4 * h_channel, 4 * h_channel, (3, 3), 1, (1, 1))

        self.conv4_1 = self._gen_conv_block(4 * h_channel, 8 * h_channel, (3, 3), 2, (1, 1))
        self.conv4_2 = self._gen_conv_block(8 * h_channel, 8 * h_channel, (3, 3), 1, (1, 1))

        self.out_conv = self._gen_conv_block(8 * h_channel, h_channel, (3, 3), 1, (1, 1))
        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.conv4_2,
            self.out_conv
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
            nn.LeakyReLU()
        )


class PoseEncoder(Encoder):
    def __init__(self, dim=10, heatmap_std=0.1, h_channel=64, heatmap_size=16):
        """

        Args:
            dim (int): Num of keypoints
        """
        super(PoseEncoder, self).__init__(h_channel)
        self.final_conv = nn.Conv2d(h_channel, dim, (3, 3), 1, (1, 1))
        self.heatmap = HeatMap(heatmap_std, (heatmap_size, heatmap_size))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = nn.functional.leaky_relu(self.final_conv(x))
        heatmap, coord = self.heatmap(x)
        return heatmap, coord


class Generator(nn.Module):
    """"""

    def __init__(self, map_size=[16, 16], channels=64 + 10, h_channel=64):
        super(Generator, self).__init__()
        self.conv1_1 = self._gen_conv_block(channels, 8 * h_channel, (3, 3), 1, (1, 1))
        self.conv1_2 = self._gen_conv_block(8 * h_channel, 8 * h_channel, (3, 3), 1, (1, 1))

        map_size = [2 * s for s in map_size]
        self.upsample1 = nn.Upsample(map_size)
        self.conv2_1 = self._gen_conv_block(8 * h_channel, 4 * h_channel, (3, 3), 1, (1, 1))
        self.conv2_2 = self._gen_conv_block(4 * h_channel, 4 * h_channel, (3, 3), 1, (1, 1))

        map_size = [2 * s for s in map_size]
        self.upsample2 = nn.Upsample(map_size)
        self.conv3_1 = self._gen_conv_block(4 * h_channel, 2 * h_channel, (3, 3), 1, (1, 1))
        self.conv3_2 = self._gen_conv_block(2 * h_channel, 2 * h_channel, (3, 3), 1, (1, 1))

        map_size = [2 * s for s in map_size]
        self.upsample3 = nn.Upsample(map_size)
        self.conv4_1 = self._gen_conv_block(2 * h_channel, h_channel, (3, 3), 1, (1, 1))
        self.conv4_2 = self._gen_conv_block(h_channel, h_channel, (3, 3), 1, (1, 1))

        self.final_conv = nn.Conv2d(h_channel, 3, (3, 3), 1, (1, 1))
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
        # return x
        return nn.functional.leaky_relu(x)  # + 1) / 2.0

    @staticmethod
    def _gen_conv_block(inc, outc, size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=size, stride=stride, padding=padding),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU()
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

        # Calculate weighted position of joint(0~1)
        x_mean = get_gaussian_mean(x, h_axis, w_axis)  # BxC
        y_mean = get_gaussian_mean(x, w_axis, h_axis)  # BxC

        coord = torch.stack([y_mean, x_mean], dim=2)

        x_mean = x_mean.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.out_h, self.out_w)
        y_mean = y_mean.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.out_h, self.out_w)

        x_ind = torch.tensor(torch.linspace(0, 1, self.out_h)).unsqueeze(-1).repeat(batch, channel, 1, self.out_w).to(
            x.device)
        y_ind = torch.tensor(torch.linspace(0, 1, self.out_w)).unsqueeze(0).repeat(batch, channel, self.out_h, 1).to(
            x.device)

        dist = (x_ind - x_mean) ** 2 + (y_ind - y_mean) ** 2

        res = torch.exp(-(dist + 1e-6).sqrt_() / 2 * self.std ** 2)
        return res, coord


if __name__ == '__main__':
    t = IMM(10, 0.1)
    x = torch.randn(10, 3, 128, 128)
    # x: the original image
    y = torch.randn(10, 3, 128, 128)
    # y: the warped image

    tmp = t(x, y)
