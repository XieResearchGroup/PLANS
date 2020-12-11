import random
import json
from collections import defaultdict

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MolFromSmiles

from ._base_loader import _BaseDataLoader


def _myShuffle(x, *s):
    x[slice(*s)] = random.sample(x[slice(*s)], len(x[slice(*s)]))


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
        """Load the json file
    
        Returns: 
            The data in json file. dict or list in most cases.
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

    def _scaffold_split(self, ratio, shuffle, nl_symbol):
        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = defaultdict(list)
        n_samples = 0
        for i, data in enumerate(self.data_list):
            if nl_symbol in data[2]:  # partially labeled data
                continue
            smiles = data[0]
            if MolFromSmiles(smiles) is None:
                continue
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smiles, includeChirality=True
            )
            all_scaffolds[scaffold].append(i)
            n_samples += 1

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set
            for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
            )
        ]

        # shuffle the order of the sets that have less than 5 members
        # make label distribution more even
        if shuffle:
            for i, scaffold_set in enumerate(all_scaffold_sets):
                if len(scaffold_set) <= 5:
                    start_point = i
                    break
            _myShuffle(all_scaffold_sets, start_point, None)

        # get train, valid, and test indices
        train_cutoff = ratio * n_samples
        train_idx, test_idx = [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(test_idx))) == 0

        return train_idx, test_idx

    def load_data(
        self, ratio=0.9, shuffle=True, nl_symbol="_", scaffold_splitting=False
    ):
        """Load the training and testing sets.
        Args:
            ratio (float): the ratio between training and testing set.
            shuffle (bool): if shuffle the indices.
            nl_symbol: the symbol denoting data with no label in the dataset.
            scaffold_splitting (bool): Use scaffold splitting. (ICLR2020, ContextPred)
        
        Returns:
            data (list): [x_train, y_train, x_test, y_test]
        """
        if scaffold_splitting:
            training_indices, testing_indices = self._scaffold_split(
                ratio, shuffle, nl_symbol
            )
        else:
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
        labels = list()
        for data in self.data_list:
            if nl_symbol not in data[2]:
                continue
            unlabeled.append(np.array(data[1]))
            labels.append(data[2])
        return np.stack(unlabeled), labels
