import warnings

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from IMMmodel import IMM
from datasets import CelebA
from utils.tps import TPS_Twice
from utils.transformers import Rescale, ToTensor

warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    model = IMM(dim=10, heatmap_std=0.1, h_channel=32)
    model.load_state_dict(torch.load('experiment4/eval_best_model_CelebA.pt'))
    model.eval()

    data_set = CelebA(transform=Compose(
        [Rescale([128, 128]), ToTensor()]))
    trainset, testset = data_set(100000, 10000)
    data_loader = DataLoader(dataset=trainset, batch_size=10, drop_last=True,
                             shuffle=True)

    sample = next(iter(data_loader))
    tps_transform = TPS_Twice(5, 0.05)
    image = sample['image']
    x1, mask1, x2, mask2 = tps_transform(image)
    recovered_x2, _ = model(x1, x1)

    plt.subplot(1, 3, 1)
    plt.imshow(x1[1].permute(1, 2, 0).detach().numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(x2[1].permute(1, 2, 0).detach().numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(recovered_x2[1].permute(1, 2, 0).detach().numpy())
    plt.show()


    def l2_reconstruction_loss(x, x_, loss_mask=None):
        loss = (x - x_) ** 2
        if loss_mask is not None:
            loss = loss * loss_mask
        return torch.mean(loss)


    l2_reconstruction_loss(x2, recovered_x2, mask2)
    pose, cord = model.pose_encoder(x1)
    plt.figure(figsize=(20, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x1[i].permute(1, 2, 0).detach().numpy())
        plt.scatter(cord[i][:, 0].detach().numpy() * 128, cord[i][:, 1].detach().numpy() * 128)
    plt.show()

x = x1
for layer in model.pose_encoder.conv_layers:
    x = layer(x)
x = torch.nn.functional.leaky_relu(model.pose_encoder.final_conv(x))

hm, c = model.pose_encoder.heatmap(x)
plt.figure(figsize=(20, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x[0][i].detach().numpy())
    plt.scatter(cord[0][i, 0].detach().numpy() * 16, cord[0][i, 1].detach().numpy() * 16)
plt.show()
