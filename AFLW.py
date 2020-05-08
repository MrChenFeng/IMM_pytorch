import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np


class AFLW(Dataset):
    """
    AFLW
    """

    def __init__(self, data_root='/home/chen/Datasets/Faces and human pose/AFLW/aflw/data', is_train=True,
                 transform=None):
        # specify annotation file for dataset
        if is_train:
            self.csv_file = data_root + '/face_landmarks_aflw_train.csv'
        else:
            self.csv_file = data_root + '/face_landmarks_aflw_test.csv'

        self.is_train = is_train
        self.transform = transform
        self.data_root = data_root
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

        image = Image.open(image_path).convert("RGB")

        top = center_h - box_size / 2.0
        bottom = center_h + box_size / 2.0
        left = center_w - box_size / 2.0
        right = center_w + box_size / 2.0
        image = image.crop([left, top, right, bottom])
        label = pts - [left, top]
        sample = {'image': image, 'label': label}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


if __name__ == '__main__':
    from transformers import Rescale, Normalize, ToTensor
    from torchvision.transforms import Compose

    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t = AFLW(transform=Compose(
        [Rescale((224, 224), is_labeled=True), ToTensor(is_labeled=True)]))  # , Normalize(mean, std,is_labeled=True)]))
    from torch.utils.data import DataLoader
    data = DataLoader(t, 10)
    import matplotlib.pyplot as plt

    tmp = next(iter(data))
    img = tmp['image']
    landmark = tmp['label']


    def display(image, label):
        if image.shape[0] == 3:
            image = image.permute(1, 2, 0)
        plt.imshow(image)
        plt.scatter(label[:, 0], label[:, 1], c='r', marker='o')
        # plt.scatter(bbox[:, 0], bbox[:, 1], c='r', marker='o')
        plt.show()


    from utils import gaussian_like_function

    t = gaussian_like_function(landmark, 224, 224)
    display(img[0], landmark[0]*224)
    display(t[0][0],landmark[0]*224)
