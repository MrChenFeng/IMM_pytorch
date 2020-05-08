# Transformers
# - Crop: Cropping image patches to given size (Receive PIL image)
# - Rescale: Resize image to square size by PIL ANTIALIAS operation (Receive PIL image, landmarks will be rescaled to [0~1])
# - ToTensor: Convert to image tensor (Receive PIL image)
# - Rotate: Rotate image patch by a random degree(90,180,270,360) (Receive ndarray or tensor)
# - Normalize: Normalize image matrix (Receive ndarray or tensor)

import torch
from torchvision import transforms
from PIL import Image
import numpy as np

'''
Transformations for landmarks labelled datasets.

According to the training goal, the dataset will be converted
Dataset should be dict{'image': imgdata, 'label': labeldata}
'''


class BaseTransformer(object):
    """Base transformer for all transformers.

    Args:
        is_labeled(bool): The annotation status of given dataset.
    """

    def __init__(self, is_labeled=False):
        self.is_labeled = is_labeled


class Crop(BaseTransformer):
    """Crop the image to a given bounding box.

    Args:
        output_box (tuple or int): Desired output box. If int, square crop
            is made.
    """

    def __init__(self, output_box, is_labeled=False):
        super(Crop, self).__init__(is_labeled)
        self.output_size = output_box

    def __call__(self, sample):
        image = sample['image']

        left = self.output_size[0]
        top = self.output_size[1]
        right = self.output_size[2]
        bottom = self.output_size[3]
        image = image[bottom: top, left: right]
        if self.is_labeled:
            label = sample['label']
            label = label - [left, top]
            return {'image': image, 'labels': label}
        return {'image': image}


class Rescale(BaseTransformer):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or list): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, is_labeled=False):
        super(Rescale, self).__init__(is_labeled)
        assert isinstance(output_size, (tuple, list))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        w, h = image.size

        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)

        image = image.resize((new_h, new_w), Image.BILINEAR)
        if self.is_labeled:
            # h and w are swapped for label because for images,
            # x and y axes are axis 1 and 0 respectively
            label = sample['label']
            label = label * [1.0 / w, 1.0 / h]
            return {'image': image, 'label': label}
        return {'image': image}


class ToTensor(BaseTransformer):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, is_labeled=False):
        super(ToTensor, self).__init__(is_labeled)

    def __call__(self, sample):
        image = sample['image']
        image = np.array(image)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        if self.is_labeled:
            # h and w are swapped for label because for images,
            # x and y axes are axis 1 and 0 respectively
            label = sample['label']
            return {'image': torch.from_numpy(image).float().div(255),
                    'label': torch.from_numpy(label)}
        return {'image': torch.from_numpy(image).float().div(255)}


class Normalize(BaseTransformer):
    """Normalize image matrix by given mean and std vector.

    Args:
        mean(array): mean vector
        std(array: std vector
    """

    def __init__(self, mean, std, is_labeled=False):
        super(Normalize, self).__init__(is_labeled)
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        image = sample['image']
        image = self.normalize(image)
        if self.is_labeled:
            label = sample['label']
            return {'image': image,
                    'label': label}
        return {'image': image}


class Rotate(BaseTransformer):
    """Rotate images by given degree. Samples should be save in dict.

    To be continued: how to rotate labels?
    """

    def __init__(self, is_labeled=False):
        super(Rotate, self).__init__(is_labeled)

    def __call__(self, sample):
        image = sample['image']
        rotation = torch.randint(0, 4, [1]).item()

        rotated_image = torch.rot90(image, rotation, [1, 2])
        # Error here. Rot will change the dimension!!!!!!!
        # label = torch.rot90(label)
        if self.is_labeled:
            label = sample['label']
            return {'image': image, 'rotated_image': rotated_image, 'label': label,'nonrotation':0, 'rotation': rotation}

        return {'image': image, 'rotated_image': rotated_image, 'rotation': rotation}
