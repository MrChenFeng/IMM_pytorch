import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn as nn

class AFLW(Dataset):
    """
    AFLW
    """

    def __init__(self, dataroot='/home/lab/datasets/AFLW', is_train=True, transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = '/home/chen/General_Datasets/Face-annotations/data/aflw/face_landmarks_aflw_train.csv'
        else:
            self.csv_file = '/home/chen/General_Datasets/Face-annotations/data/aflw/face_landmarks_aflw_test.csv'

        self.is_train = is_train
        self.transform = transform
        self.data_root = dataroot
        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]
        box_size = self.landmarks_frame.iloc[idx, 2]

        center_w = self.landmarks_frame.iloc[idx, 3]
        center_h = self.landmarks_frame.iloc[idx, 4]
        # center = torch.Tensor([center_w, center_h])

        pts = self.landmarks_frame.iloc[idx, 5:].values
        pts = pts.astype("float").reshape(-1, 2)

        img = Image.open(image_path).convert("RGB")

        top = center_h - box_size / 2.0
        bottom = center_h + box_size / 2.0
        left = center_w - box_size / 2.0
        right = center_w + box_size / 2.0
        img = img.crop([left, top, right, bottom])
        landmarks = pts - [left, top]
        sample = {'image': img, 'landmarks': landmarks}
        if self.transform is not None:
            sample = self.transform(sample)
        # meta = {
        #     "index": idx,
        #     "center": center,
        #     "rescale": scale,
        #     "pts": torch.Tensor(pts),
        #     # "tpts": tpts,
        #     "bbox": bbox,
        # }

        return sample


if __name__ == '__main__':
    from transformers import Rescale, Normalize, ToTensor
    from torchvision.transforms import Compose

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t = AFLW(transform=Compose([Rescale((224, 224)), ToTensor(), Normalize(mean, std)]))
    tmp = t[1000]
    import matplotlib.pyplot as plt

    img = tmp['image']
    landmark = tmp['landmarks']


    def display(image, label):
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        plt.imshow(image)
        plt.scatter(label[:, 0], label[:, 1], c='r', marker='o')
        # plt.scatter(bbox[:, 0], bbox[:, 1], c='r', marker='o')
        plt.show()

    display(img, landmark)
