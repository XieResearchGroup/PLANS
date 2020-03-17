import random

from h5py import File
import numpy as np


class HDF5Loader(File):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__field = None
        self.__index = 0

    def set_dataset(self, key, shuffle=False, infinite=False):
        r""" Set the default dataset.
        key (str): the dataset path in the hdf5 file
        shuffle (bool): whether to shuffle the dataset
        infinite (bool): whether to reset the dataset when the iteration
            reaches the end of the dataset

        return: None
        """
        self.__field = key
        self.__indices = list(range(self[key].shape[0]))
        self.__shuffle = shuffle
        self.__infinite = infinite
        if shuffle:
            random.shuffle(self.__indices)

    def _check_field(self):
        if self.__field is None:
            raise AttributeError("Set the dataset to iter from first by "
                                 "calling self.set_dataset() method.")

    def __iter__(self):
        self._check_field()
        return self

    def _get_value(self):
        value = self[self.__field][self.__indices[self.__index]]
        self.__index += 1
        return value

    def __next__(self):
        self._check_field()
        try:
            value = self._get_value()
        except IndexError:
            if self.__infinite:
                self.__index = 0
                if self.__shuffle:
                    random.shuffle(self.__indices)
                value = self._get_value()
            else:
                raise StopIteration
        return value

    def set_batch_size(self, batch_size):
        self.__batch_size = batch_size

    @property
    def steps(self):
        try:
            self._check_field()
            return int(self[self.__field].shape[0] // self.__batch_size)
        except AttributeError as e:
            raise e("Set the batch size with "
                    "self.set_batch_size(batch_size) or call "
                    "self.batch_loader(batch_size) before getting the steps.")

    def batch_loader(self, batch_size=None):
        r""" Mini batch generator. The batch_size argument will overwrite the
        batch size set by the set_batch_size() method. Leave it as default if
        you do not want to change the batch size.
        batch_size (int): mini batch size.

        yield (numpy.array): mini batch.
        """
        self._check_field()
        if batch_size is not None:
            self.__batch_size = batch_size
        try:
            batch = np.empty((self.__batch_size, self[self.__field].shape[1]),
                             dtype=self[self.__field].dtype)
        # the dataset does not have dimension value for the features
        except IndexError:
            batch = np.empty((self.__batch_size, 1),
                             dtype=self[self.__field].dtype)
        while True:
            index = 0
            while index < self.__batch_size:
                batch[index] = next(self)
                index += 1
            yield batch
