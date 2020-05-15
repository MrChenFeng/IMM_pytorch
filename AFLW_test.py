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
    model.load_state_dict(torch.load('Celeba_originaldata_experiment2/checkpoint_model_CelebA.pt'))
    model.eval()

    data_set = CelebA(transform=Compose(
        [Rescale([128, 128]), ToTensor()]))
    trainset, testset = data_set(10000, 1000)
    data_loader = DataLoader(dataset=testset, batch_size=10, drop_last=True,
                             shuffle=True)

    # train_set = AFLW(is_train=True, transform=Compose(
    #     [Rescale([128,128], is_labeled=True), ToTensor(is_labeled=True)]))
    # data_loader = DataLoader(dataset=train_set, batch_size=10, drop_last=True,
    #                                shuffle=True)

    sample = next(iter(data_loader))
    tps_transform = TPS_Twice(5, 0.05, 0.05)
    image = sample['image']
    x1, mask1, x2, mask2 = tps_transform(image)
    recovered_x2, cord = model(x1, x2)

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


    print(l2_reconstruction_loss(x2, recovered_x2, mask2))
    pose, cord = model.pose_encoder(x2)
    plt.imshow(x2[0].permute(1, 2, 0).detach().numpy())
    plt.scatter(cord[0][:, 0].detach().numpy() * 128, cord[0][:, 1].detach().numpy() * 128, c=c)
    plt.show()

    pose, cord = model.pose_encoder(x1)
    plt.imshow(x1[0].permute(1, 2, 0).detach().numpy())
    plt.scatter(cord[0][:, 0].detach().numpy() * 128, cord[0][:, 1].detach().numpy() * 128, c=c)
    plt.show()

    pose, cord = model.pose_encoder(sample['image'])
    plt.imshow(recovered_x2[0].permute(1, 2, 0).detach().numpy())
    plt.scatter(cord[0][:, 0].detach().numpy() * 128, cord[0][:, 1].detach().numpy() * 128, c=c)
    plt.show()

    plt.figure(figsize=(20, 5))
    import numpy as np

    c = np.linspace(0, 255, 10) / 255
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample['image'][i].permute(1, 2, 0).detach().numpy())
        plt.scatter(cord[i][:, 0].detach().numpy() * 128, cord[i][:, 1].detach().numpy() * 128, c=c)
    plt.show()

x = sample['image'][0]
for layer in model.pose_encoder.conv_layers:
    x = layer(x)
# x = model.pose_encoder.final_conv(x)

hm, cord = model.pose_encoder.heatmap(x)

plt.figure(figsize=(20, 5))
for i in range(10):
    plt.subplot(2, 10, i + 1)
    t = (x[0][i] - x[0][i].min()) / (x[0][i].max() - x[0][i].min())
    plt.imshow(t.detach().numpy(), cmap='gray')  # ,vmin=x[0][i].min(), vmax=x[0][i].max())
    plt.scatter(cord[0][i, 0].unsqueeze(0).detach().numpy() * 16, cord[0][i, 1].unsqueeze(0).detach().numpy() * 16,
                c='r', marker='*', cmap='RGB')
plt.subplot(2,5)
plt.show()


def generate_heatmap(x_mean, y_mean):
    x_mean = x_mean.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 16, 16)
    y_mean = y_mean.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 16, 16)

    x_ind = torch.tensor(torch.linspace(0, 1, 16)).unsqueeze(-1).repeat(10, 10, 1, 16).to(
        x.device)
    y_ind = torch.tensor(torch.linspace(0, 1, 16)).unsqueeze(0).repeat(10, 10, 16, 1).to(
        x.device)

    dist = (x_ind - x_mean) ** 2 + (y_ind - y_mean) ** 2

    res = torch.exp(-(dist + 1e-6).sqrt_() / (2 * 0.1 ** 2))
    return res


content1 = model.content_encoder(x1)
content2 = model.content_encoder(x2)
pose2, cord2 = model.pose_encoder(x2)
pose1, cord1 = model.pose_encoder(x1)
hm_fake = generate_heatmap(torch.rand(10, 10), torch.rand(10, 10))


#2+2
testsingle(content2,pose2,2,3)
#1+1
testsingle(content1,pose1,2,3)
#2+1
testsingle(content2,pose1,2,3)
#1+2
testsingle(content1,pose2,2,3)

i = 1
j=2
code = torch.cat((content2[i].unsqueeze(0), pose2[j].unsqueeze(0)), dim=1)
recovered1 = model.generator(code)

plt.subplot(1, 3, 1)
plt.imshow(x2[i].permute(1, 2, 0).detach().numpy())
plt.subplot(1, 3, 2)
plt.imshow(x2[j].permute(1, 2, 0).detach().numpy())
plt.subplot(1, 3, 3)
plt.imshow(recovered1[0].permute(1, 2, 0).detach().numpy())
plt.show()

#2+2
test(content2,pose2)
#1+1
test(content1,pose1)
#2+1
test(content2,pose1)
#1+2
test(content1,pose2)


def test(content, pose):
    code = torch.cat((content, pose), dim=1)
    recovered1 = model.generator(code)

    plt.subplot(1, 3, 1)
    plt.imshow(x1[0].permute(1, 2, 0).detach().numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(x2[0].permute(1, 2, 0).detach().numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(recovered1[0].permute(1, 2, 0).detach().numpy())
    plt.show()

hm2, cord2 = model.pose_encoder.heatmap(hm)
plt.figure(figsize=(20, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(hm2[0][i].detach().numpy() * 255)
    plt.scatter(cord2[0][i, 0].detach().numpy() * 16, cord2[0][i, 1].detach().numpy() * 16, c='r')
plt.show()

plt.imshow(x2[0].permute(1, 2, 0).detach().numpy())
plt.scatter(cord[0][:, 0].detach().numpy() * 128, cord[0][:, 1].detach().numpy() * 128, c=c)
plt.show()

#  test content code


import torchvision

grid_img = torchvision.utils.make_grid(content1, nrow=5)
plt.imshow(grid_img[0].detach().numpy())
plt.show()

grid_img = torchvision.utils.make_grid(content2, nrow=5)
plt.imshow(grid_img[0].detach().numpy())
plt.show()

mse = torch.nn.MSELoss()
mse(content1, content2)
