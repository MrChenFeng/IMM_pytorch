from tps import TPS_Twice
from aflw import AFLW
from transformers import Rescale, ToTensor
from torchvision.transforms import Compose
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

# Running Settings%
batch = 100
size = [128, 128]
dataset = AFLW
epochs = 500
njoints = 10
lr = 0.001

train_set = dataset(is_train=False, transform=Compose([Rescale(size), ToTensor()]))
train_loader = DataLoader(dataset=train_set, batch_size=batch, drop_last=True, shuffle=True)

model = IMM(dim=njoints).cuda()
model.load_state_dict(torch.load('results/MSE_Experiment_2020-05-06-21-02-50/100 epochs_finalloss30.2097247838974.pt'))
optimizer = Adam(
    [{'params': model.parameters(), 'lr': lr}],
    weight_decay=5e-4)  # , {'params': hmap.parameters()},{'params': pmap.parameters()}])
scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=10, verbose=True)

#####################################How to measure the recovered image####################################
sim_metric = nn.MSELoss().cuda()
# sim_metric = LossFunc().cuda()
# Tobe done: vggconcept loss
################################################################################################
