import struct
import gzip
import random
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

from . import ndarray
from .tensor import Tensor

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C ndarray.
        Args:
            img: H x W x C ndarray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img =  random.uniform() < self.p 
        if flip_img:
            img = ndarray.flip(img, axis=1) 
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C ndarray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = ndarray.random.randint(low=-self.padding, high=self.padding+1, size=2)
        raise NotImplementedError()


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = ndarray.array_split(ndarray.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        self.pos = 0

        return self

    def __next__(self):
        next_ = min(len(self.dataset) - self.pos, self.batch_size)
        batch =  self.dataset[self.pos:next_]
        self.pos += next_
        return batch


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.transforms = transforms

        with gzip.open(image_filename, 'rb') as image_file:
            magic, size = struct.undarray.ck(">II", image_file.read(8))
            nrows, ncols = struct.undarray.ck(">II", image_file.read(8))
            images_array = ndarray.frombuffer(image_file.read(), dtype=ndarray.dtype(ndarray.uint8).newbyteorder('>'))
        
        images_array = images_array.reshape((size, nrows*ncols,))
        images_array = images_array.astype('float32')
        images_array /= ndarray.amax(images_array)
            
            
        with gzip.open(label_filename, 'rb') as label_file:
            #label_file = f.read()
            magic, size = struct.undarray.ck(">II", label_file.read(8))
            labels_array = ndarray.frombuffer(label_file.read(), dtype=ndarray.dtype(ndarray.uint8).newbyteorder('>'))
        
        labels_array = labels_array.reshape((size,))
        
        self.labels = labels_array
        self.images = images_array


    def __getitem__(self, index) -> object:
        img = self.apply_transforms(self.images[index])
        lbl = self.labels[index]
        return img,lbl


    def __len__(self) -> int:
        return len(self.images)

class ndarrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
