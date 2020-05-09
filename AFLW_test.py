from datasets import AFLW
from datasets import CelebA
from utils.transformers import Rescale, ToTensor
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from IMMmodel import IMM
import warnings
import matplotlib.pyplot as plt
from utils.tps import TPS_Twice
import torch

warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    model = IMM(dim=10, heatmap_std=0.1)
    model.load_state_dict(torch.load('model_AFLW.pt'))
    model.eval()

    data_set = AFLW(is_train=True, transform=Compose(
        [Rescale([128, 128]), ToTensor()]))
    data_loader = DataLoader(dataset=data_set, batch_size=10, drop_last=True,
                             shuffle=True)

    sample = next(iter(data_loader))
    tps_transform = TPS_Twice(5, 0.05)
    image = sample['image']
    x1, mask1, x2, mask2 = tps_transform(image)
    recovered_x2, _ = model(x1, x2)

    plt.subplot(1, 3, 1)
    plt.imshow(x1[0].permute(1, 2, 0).detach().numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(x2[0].permute(1, 2, 0).detach().numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(recovered_x2[0].permute(1, 2, 0).detach().numpy())
    plt.show()


    def l2_reconstruction_loss(x, x_, loss_mask=None):
        loss = (x - x_) ** 2
        if loss_mask is not None:
            loss = loss * loss_mask
        return torch.mean(loss)

    l2_reconstruction_loss(x2, recovered_x2, mask2)
    pose, cord = model.pose_encoder(image)
    plt.figure(figsize=(20, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample['image'][i].permute(1, 2, 0))
        plt.scatter(cord[i][:, 0].detach().numpy() * 128, cord[i][:, 1].detach().numpy() * 128)
    plt.show()
