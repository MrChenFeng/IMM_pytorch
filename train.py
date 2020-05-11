# -*- coding: utf-8 -*-
# from apex import amp
import statistics as stats
import warnings

import torch
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose
from tqdm import tqdm

from IMMmodel import IMM
from configs.config import config
from datasets import AFLW
from datasets import CelebA
from utils.tps import TPS_Twice
from utils.transformers import Rescale, ToTensor

warnings.filterwarnings("ignore", category=UserWarning)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Running Settings%
def l2_reconstruction_loss(x, x_, loss_mask=None):
    loss = (x - x_) ** 2
    if loss_mask is not None:
        loss = loss * loss_mask
    return torch.mean(loss)


class Trainer(object):
    def __init__(self, config):
        self.config = config

        if self.config.dataset == 'AFLW':
            self.dataset = AFLW
        elif self.config.dataset == 'CelebA':
            self.dataset = CelebA

        if self.config.split == 'listed':
            self._load_list_split()
        elif self.config.split == 'random':
            self._load_random_split()

        self.train_step = 0
        self.test_step = 0

        # self.__init_logger()
        self._load_model()
        self.writer = SummaryWriter(self.config.run_id)
        self._init_TPS()
        self.metric = l2_reconstruction_loss
        self.eval_loss = float("inf")

    def _load_list_split(self):
        train_set = self.dataset(is_train=True, transform=Compose(
            [Rescale([self.config.data_rescale_height, self.config.data_rescale_width]), ToTensor()]))
        self.train_loader = DataLoader(dataset=train_set, batch_size=self.config.batch_size, drop_last=True,
                                       shuffle=True)
        test_set = self.dataset(is_train=False, transform=Compose(
            [Rescale([self.config.data_rescale_height, self.config.data_rescale_width]), ToTensor()]))
        self.test_loader = DataLoader(dataset=test_set, batch_size=self.config.batch_size, drop_last=True,
                                      shuffle=True)

    def _load_random_split(self):
        data_set = self.dataset(transform=Compose(
            [Rescale([self.config.data_rescale_height, self.config.data_rescale_width]), ToTensor()]))
        trainlen = int(self.config.trainratio * len(data_set))
        testlen = int(self.config.testratio * len(data_set))
        train_set, test_set = data_set(trainlen, testlen)
        self.train_loader = DataLoader(dataset=train_set, batch_size=self.config.batch_size, drop_last=True,
                                       shuffle=True)
        self.test_loader = DataLoader(dataset=test_set, batch_size=self.config.batch_size, drop_last=True,
                                      shuffle=True)

    def _load_model(self):
        self.model = IMM(dim=self.config.num_keypoints, heatmap_std=self.config.heatmap_std).to(self.config.device)
        # if self.config.pretrained is not None:
        #     self.model.load_state_dict(torch.load(self.config.pretrained))
        self.optimizer = Adam([{'params': self.model.parameters(), 'lr': self.config.lr}])  # , {'params': hmap.parameters()},{'params': pmap.parameters()}])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=0, verbose=True)
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")

    def _init_TPS(self):
        self.tps_transform = TPS_Twice(self.config.tps_control_pts, self.config.tps_variance, self.config.max_rot)

    def train(self, epoch):
        cur_loss = 0
        self.model.train()
        log_loss = []
        print(f'------------------------Epoch {epoch + 1} started!-----------------------')
        batch = tqdm(self.train_loader, total=len(self.train_loader), position=0, leave=True, ascii=True)
        for i, sample in enumerate(batch):
            self.optimizer.zero_grad()
            image = sample['image'].to(self.config.device)  # BxCxHxW
            x1, _, x2, loss_mask = self.tps_transform(image)
            recovered_x2, _ = self.model(x1, x2)
            loss = self.metric(recovered_x2, x2, loss_mask)
            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_
            loss.backward()
            #clip_grad_norm_(self.model.parameters(), 1)
            cur_loss += float(loss)
            if len(log_loss) > 20:
                log_loss.pop(0)
            log_loss.append(loss.item())
            self.optimizer.step()
            self.writer.add_scalar('Train_loss', loss.item(), global_step=self.train_step)
            self.train_step += 1
            batch.set_description(f'Iters: {i + 1} Train Loss: {stats.mean(log_loss)}')
        print(f'--- Epoch {epoch + 1} Train Loss: {cur_loss} ---')
        self.scheduler.step(cur_loss)

    def eval(self, epoch):
        cur_loss = 0
        log_loss = []
        self.model.eval()
        batch = tqdm(self.test_loader, total=len(self.test_loader), position=0, leave=True, ascii=True)
        for i, sample in enumerate(batch):
            image = sample['image'].to(self.config.device)  # BxCxHxW
            x1, _, x2, loss_mask = self.tps_transform(image)
            recovered_x2, _ = self.model(x1, x2)
            loss = self.metric(recovered_x2, x2, loss_mask)
            cur_loss += float(loss)
            if len(log_loss) > 20:
                log_loss.pop(0)
            log_loss.append(loss.item())
            self.writer.add_scalar('Test_loss', loss.item(), global_step=self.test_step)
            self.test_step += 1
            batch.set_description(f'Iters: {i + 1} Eval Loss: {stats.mean(log_loss)}')
        print(f'--- Minimum Eval Loss: {self.eval_loss} --- Epoch {epoch + 1} Loss: {cur_loss} ---')
        if self.eval_loss > cur_loss:
            self.eval_loss = cur_loss
            torch.save(self.model.state_dict(), self.config.run_id + f'/model_{self.config.dataset}.pt')


if __name__ == '__main__':
    configs = config()
    trainer = Trainer(configs)
    # print(len(trainer.test_loader))
    print('Configs:', configs)
    for j in range(configs.epochs):
        trainer.train(j)
        trainer.eval(j)
