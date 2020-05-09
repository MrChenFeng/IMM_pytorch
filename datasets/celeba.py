from torch.utils.data import Dataset, Subset
from os.path import join
from PIL import Image
import os
import numpy as np


class Base(Dataset):
    """CelebA dataset."""

    def __init__(self, root_dir='/home/lab/datasets/CelebA/Img/img_faces_celeba', transform=None):
        """
        Args:
            root_dir (string):       Directory with all the images.
            transform (callable,
                       optional):    Optional transform to be applied
                                     on a sample.
        """
        self.root_dir = root_dir
        self.filelist = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_name = join(self.root_dir, self.filelist[idx])
        image = Image.open(img_name).convert("RGB")

        sample = {'image': image}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class CelebA(object):
    def __init__(self, transform=None):
        self.dataset = Base(transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __call__(self, trainlen, testlen=None):
        imagelist = self.dataset.filelist
        np.random.seed(1)
        ind = np.random.permutation(len(imagelist))
        trainset = Subset(self.dataset, ind[:trainlen])
        if testlen == None:
            testset = Subset(self.dataset, ind[trainlen:])
        else:
            testset = Subset(self.dataset, ind[trainlen:(trainlen + testlen)])
        return trainset, testset


if __name__ == '__main__':
    from utils.transformers import Rescale, ToTensor
    from torchvision.transforms import Compose

    celeba_dataset = CelebA(Compose([Rescale([128, 128]), ToTensor()]))
    trainset, testset = celeba_dataset(100000, 50000)
    print(len(trainset))
    print(len(testset))
    import matplotlib.pyplot as plt

    plt.imshow(trainset[100]['image'].permute(1, 2, 0))
    plt.show()
