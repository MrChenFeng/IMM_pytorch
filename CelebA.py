from torch.utils.data import Dataset
from os.path import join, exists, isdir
from PIL import Image

class CelebA(Dataset):
    """CelebA dataset."""

    def __init__(self, root_dir, train_mode = True, transform=None):
        """
        Args:
            root_dir (string):       Directory with all the images and train/test list.
            train_mode (string):           Train set or Test set.
            transform (callable,
                       optional):    Optional transform to be applied
                                     on a sample.
        """
        if not exists(root_dir):
            err = 'Celeba aligned images directory is not found: %s' % root_dir
            raise FileNotFoundError(err)
        if not isdir(root_dir):
            err = '%s must be a directory with aligned images' % root_dir
            raise NotADirectoryError(err)

        self.root_dir = root_dir
        self.transform = transform
        if self.trainmode:
            self.filelist =

    def __len__(self):
        return len(self.filelist[self.mode])

    def __getitem__(self, idx):
        img_name = join(self.root_dir, self.filelist[idx])
        image = default_loader(img_name)

        if self.transform is not None:
            image = self.transform(image)

        return image