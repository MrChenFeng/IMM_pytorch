import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# class Crop(object):
#     """Crop image based on given bounding box.
#
#     Args:
#         bounding_box(tuple or int): Human annotated bounding boxes.
#     """
#
#     def __call__(self, sample, *args, **kwargs):
#         image, landmarks = sample['image'], sample['landmarks']
#

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        w, h = image.size

        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_h, new_w), Image.BILINEAR)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = np.array(image)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float().div(255),
                'landmarks': torch.from_numpy(landmarks)}


class Normalize(object):
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = self.normalize(image)

        return {'image': image,
                'landmarks': landmarks}
