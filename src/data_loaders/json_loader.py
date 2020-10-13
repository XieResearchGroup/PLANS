import random
import json

import numpy as np

from ._base_loader import _BaseDataLoader


class JsonLoader(_BaseDataLoader):
    def __init__(self, path, *_, rand_seed=None, **kwargs):
        """
        path: path to the csv file
        kwargs: keyword arguments for the pandas.read_csv function
        """
        super().__init__(path)
        self.kwargs = kwargs
        if rand_seed is not None:
            random.seed(rand_seed)

    @property
    def data_list(self):
        """Load the csv file into pandas dataframe when first requested
        ================================================================
        return: pandas dataframe
        """
        try:
            return self._data_list
        except AttributeError:
            with open(self.path, "r") as f:
                self._data_list = json.load(f)
            return self._data_list

    def _split(self, ratio, shuffle, nl_symbol):
        indices = list()
        for idx, data in enumerate(self.data_list):
            if nl_symbol in data[2]:
                continue
            indices.append(idx)
        if shuffle:
            random.shuffle(indices)
        splitter = int(len(indices) * ratio)
        training_indices = indices[:splitter]
        testing_indices = indices[splitter:]
        return training_indices, testing_indices

    def load_data(self, ratio=0.9, shuffle=True, nl_symbol="_"):
        """Load the training and testing sets.
        params:
        ratio (float): the ratio between training and testing set.
        nl_symbol: the symbol denoting data with no label in the dataset.
        ================================================================
        return:
        data (list): [x_train, y_train, x_test, y_test]
        """
        training_indices, testing_indices = self._split(ratio, shuffle, nl_symbol)
        x_train, y_train, x_test, y_test = list(), list(), list(), list()
        for idx in training_indices:
            x_train.append(self.data_list[idx][1])
            y_train.append(self.data_list[idx][2])
        for idx in testing_indices:
            x_test.append(self.data_list[idx][1])
            y_test.append(self.data_list[idx][2])
        x_train = np.array(x_train, dtype=np.float)
        y_train = np.array(y_train)
        x_test = np.array(x_test, dtype=np.float)
        y_test = np.array(y_test)
        return x_train, y_train, x_test, y_test

    def load_unlabeled(self, nl_symbol="_"):
        """Load the unlabeled data.
        params:
        nl_symbol: the symbol denoting data with no label dataset.
        ================================================================
        return:
        data (numpy array): unlabeled dataset with respect to the specified
                            label column
        """
        unlabeled = list()
        for data in self.data_list:
            if nl_symbol not in data[2]:
                continue
            unlabeled.append([np.array(data[1]), data[2]])
        return unlabeled
