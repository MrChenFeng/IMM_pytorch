import torch
import torch.nn as nn

from utils import get_gaussian_mean


class IMM(nn.Module):
    def __init__(self, dim=10, heatmap_std=0.1, in_channel=3, h_channel=32):
        """
        It should be noted all params has been fixed to Jakab 2018 paper.
        Goto the original class if params and layers need to be changed.
        Images should be rescaled to 128*128
        """
        super(IMM, self).__init__()
        self.content_encoder = ContentEncoder(in_channel, h_channel)
        self.pose_encoder = PoseEncoder(dim, heatmap_std, in_channel, h_channel)
        self.generator = Generator(channels=8*h_channel+dim, h_channel=h_channel)
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


class ContentEncoder(nn.Module):
    def __init__(self, in_channel=3, h_channel=64):
        super(ContentEncoder, self).__init__()
        self.conv1_1 = Conv_Block(in_channel, h_channel, (3, 3))
        self.conv1_2 = Conv_Block(h_channel, h_channel, (3, 3))

        self.conv2_1 = Conv_Block(h_channel, 2 * h_channel, (3, 3), downsample=True)
        self.conv2_2 = Conv_Block(2 * h_channel, 2 * h_channel, (3, 3))

        self.conv3_1 = Conv_Block(2 * h_channel, 4 * h_channel, (3, 3), downsample=True)
        self.conv3_2 = Conv_Block(4 * h_channel, 4 * h_channel, (3, 3))

        self.conv4_1 = Conv_Block(4 * h_channel, 8 * h_channel, (3, 3), downsample=True)
        self.conv4_2 = Conv_Block(8 * h_channel, 8 * h_channel, (3, 3))

        # self.out_conv = Conv_Block(8 * h_channel, h_channel, (3, 3))
        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.conv4_2
            # self.out_conv
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class PoseEncoder(nn.Module):
    def __init__(self, dim=10, heatmap_std=0.1, in_channel=3, h_channel=64, heatmap_size=16):
        """

        Args:
            dim (int): Num of keypoints
        """
        super(PoseEncoder, self).__init__()
        self.conv1_1 = Conv_Block(in_channel, h_channel, (3, 3))
        self.conv1_2 = Conv_Block(h_channel, h_channel, (3, 3))

        self.conv2_1 = Conv_Block(h_channel, 2 * h_channel, (3, 3), downsample=True)
        self.conv2_2 = Conv_Block(2 * h_channel, 2 * h_channel, (3, 3))

        self.conv3_1 = Conv_Block(2 * h_channel, 4 * h_channel, (3, 3), downsample=True)
        self.conv3_2 = Conv_Block(4 * h_channel, 4 * h_channel, (3, 3))

        self.conv4_1 = Conv_Block(4 * h_channel, 8 * h_channel, (3, 3), downsample=True)
        self.conv4_2 = Conv_Block(8 * h_channel, 8 * h_channel, (3, 3))

        self.out_conv = nn.Sequential(nn.Conv2d(8 * h_channel, dim, (1, 1)))
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
        self.heatmap = HeatMap(heatmap_std, (heatmap_size, heatmap_size))

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        heatmap, coord = self.heatmap(x)
        return heatmap, coord


class Generator(nn.Module):
    """"""

    def __init__(self, channels=64 + 10, h_channel=64):
        super(Generator, self).__init__()
        self.conv1_1 = Conv_Block(channels, 8 * h_channel, (3, 3))
        self.conv1_2 = Conv_Block(8 * h_channel, 8 * h_channel, (3, 3), upsample=True)

        self.conv2_1 = Conv_Block(8 * h_channel, 4 * h_channel, (3, 3))
        self.conv2_2 = Conv_Block(4 * h_channel, 4 * h_channel, (3, 3), upsample=True)

        self.conv3_1 = Conv_Block(4 * h_channel, 2 * h_channel, (3, 3))
        self.conv3_2 = Conv_Block(2 * h_channel, 2 * h_channel, (3, 3), upsample=True)

        self.conv4_1 = Conv_Block(2 * h_channel, h_channel, (3, 3))
        self.conv4_2 = Conv_Block(h_channel, h_channel, (3, 3))

        self.final_conv = nn.Conv2d(h_channel, 3, (3, 3), padding=[1, 1])
        self.conv_layers = nn.ModuleList([
            self.conv1_1,
            self.conv1_2,
            self.conv2_1,
            self.conv2_2,
            self.conv3_1,
            self.conv3_2,
            self.conv4_1,
            self.conv4_2,
            self.final_conv
        ])

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        # return x
        #return (nn.functional.tanh(x) + 1) / 2.0
        return nn.functional.relu(x)


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

        res = torch.exp(-(dist + 1e-6).sqrt_() / (2 * self.std ** 2))
        return res, coord


class Conv_Block(nn.Module):
    def __init__(self, inc, outc, size, downsample=False, upsample=False):
        super(Conv_Block, self).__init__()
        block = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=size),
            nn.BatchNorm2d(outc),
            nn.LeakyReLU()
        ]
        if downsample:
            block += [nn.MaxPool2d(kernel_size=2, stride=2)]
        if upsample:
            block += [nn.UpsamplingBilinear2d(scale_factor=2)]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    t = IMM(10, 0.1)
    x = torch.randn(10, 3, 128, 128)
    # x: the original image
    y = torch.randn(10, 3, 128, 128)
    # y: the warped image

    tmp = t(x, y)
