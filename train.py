from tps import TPS_Twice
from aflw import AFLW
from celeba import CelebA
from transformers import Rescale, ToTensor
from torchvision.transforms import Compose
import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from IMMmodel import IMM
from config import config
from tps import TPS_Twice
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Running Settings%
def l2_reconstruction_loss(x, x_, loss_mask=None):
    loss = (x - x_) ** 2
    if loss_mask is not None:
        loss = loss * loss_mask
    return torch.mean(loss)


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self._load_data()
        self._load_model()
        self._init_TPS()
        self.metric = l2_reconstruction_loss
        self.eval_loss = 0

    #     # self.trainround = 0
    # def count(self):
    #     self.round +=

    def _load_data(self):
        # Now only tested on AFLW and CelebA
        if self.config.dataset == 'AFLW':
            self.dataset = AFLW
        else:
            self.dataset = CelebA

        train_set = self.dataset(is_train=True, transform=Compose(
            [Rescale([self.config.data_rescale_height, self.config.data_rescale_width]), ToTensor()]))
        self.train_loader = DataLoader(dataset=train_set, batch_size=self.config.batch_size, drop_last=True,
                                       shuffle=True)
        test_set = self.dataset(is_train=False, transform=Compose(
            [Rescale([self.config.data_rescale_height, self.config.data_rescale_width]), ToTensor()]))
        self.test_loader = DataLoader(dataset=train_set, batch_size=self.config.batch_size, drop_last=True,
                                      shuffle=True)

    def _load_model(self):
        self.model = IMM(dim=self.config.num_keypoints, heatmap_std=self.config.heatmap_std).to(self.config.device)
        self.optimizer = Adam([{'params': self.model.parameters(), 'lr': self.config.lr}],
                              weight_decay=5e-4)  # , {'params': hmap.parameters()},{'params': pmap.parameters()}])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=10, verbose=True)

    def _init_TPS(self):
        self.tps_transform = TPS_Twice(self.config.tps_control_pts, self.config.tps_variance)

    def train(self):
        cur_loss = 0
        batch = tqdm(self.train_loader, total=len(self.train_loader))
        for i, sample in enumerate(batch):
            image = sample['image'].to(self.config.device)  # BxCxHxW
            x1, _, x2, loss_mask = self.tps_transform(image)
            recovered_x2, _ = self.model(x1, x2)
            loss = self.metric(x2, recovered_x2, loss_mask)
            loss.backward()
            batch.set_description(f'Iters: {i + 1} Train Loss: {loss.item()}')
            cur_loss = cur_loss + loss.item()
            self.optimizer.step()
        self.scheduler.step(cur_loss)

    def eval(self):
        cur_loss = 0
        batch = tqdm(self.test_loader, total=len(self.test_loader))
        for i, sample in enumerate(batch):
            image = sample['image'].to(self.config.device)  # BxCxHxW
            x1, _, x2, loss_mask = self.tps_transform(image)
            recovered_x2, _ = self.model(x1, x2)
            loss = self.metric(x2, recovered_x2, loss_mask)
            batch.set_description(f'Iters: {i + 1} Eval Loss: {loss.item()}')
            cur_loss = cur_loss + loss.item()
        if self.eval_loss > cur_loss:
            torch.save(self.model.state_dict('model.pt'))


if __name__ == '__main__':
    configs = config()
    trainer = Trainer(configs)
    #print(len(trainer.train_loader))
    for j in range(configs.epochs):
        print(f'-------------------------Epoch {j + 1} started!------------------------')
        trainer.train()
        trainer.eval()
        print(f'-------------------------Epoch {j + 1} ended!------------------------')
