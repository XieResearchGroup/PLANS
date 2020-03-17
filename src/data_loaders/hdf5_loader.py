import random

from h5py import File
import numpy as np


class HDF5Loader(File):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__fields = None
        self.__index = 0

    def _assert_valid(self, keys):
        # assert keys is not empty
        assert len(keys) > 0
        if len(keys) == 1:
            return
        # assert the datasets have the same amount of samples
        length = self[keys[0]].shape[0]
        for key in keys[1:]:
            assert self[key].shape[0] == length

    def set_dataset(self, *keys, shuffle=False, infinite=False):
        r""" Set the default dataset.
        key (str): the dataset path in the hdf5 file
        shuffle (bool): whether to shuffle the dataset
        infinite (bool): whether to reset the dataset when the iteration
            reaches the end of the dataset

        return: None
        """
        self._assert_valid(keys)
        self.__fields = keys
        self.__indices = list(range(self[keys[0]].shape[0]))
        self.__shuffle = shuffle
        self.__infinite = infinite
        if shuffle:
            random.shuffle(self.__indices)

    def _check_field(self):
        if self.__fields is None:
            raise AttributeError("Set the dataset to iter from first by "
                                 "calling self.set_dataset() method.")

    def __iter__(self):
        self._check_field()
        return self

    def _get_value(self):
        value = [self[field][self.__indices[self.__index]] for
                 field in self.__fields]
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
            return int(len(self.__indices) // self.__batch_size)
        except AttributeError as e:
            raise e("Set the batch size with "
                    "self.set_batch_size(batch_size) or call "
                    "self.batch_loader(batch_size) before getting the steps.")

    def batch_loader(self, batch_size=None):
        r""" Mini batch generator. The batch_size argument will overwrite the
        batch size set by the set_batch_size() method. Leave it as default if
        you do not want to change the batch size.
        batch_size (int): mini batch size.

        yield (list): list with mini batches data
        """
        self._check_field()
        if batch_size is not None:
            self.__batch_size = batch_size
        batches = list()
        for field in self.__fields:
            try:
                batch = np.empty((self.__batch_size, self[field].shape[1]),
                                 dtype=self[field].dtype)
            # the dataset does not have dimension value for the features
            except IndexError:
                batch = np.empty((self.__batch_size, 1),
                                 dtype=self[field].dtype)
            batches.append(batch)
        while True:
            index = 0
            while index < self.__batch_size:
                data = next(self)
                for i, batch in enumerate(batches):
                    batch[index] = data[i]
                index += 1
            yield batches
