import random
import itertools

import pandas as pd
from ._base_loader import _BaseDataLoader


class CVSLoader(_BaseDataLoader):

    def __init__(self, path, *_, **kwargs):
        """
        path: path to the csv file
        kwargs: keyword arguments for the pandas.read_csv function
        """
        super(CVSLoader, self).__init__(path)
        self.kwargs = kwargs

    @property
    def data_df(self):
        """ Load the csv file into pandas dataframe when first requested
        ================================================================
        return: pandas dataframe
        """
        try:
            return self._data_df
        except AttributeError:
            self._data_df = pd.read_csv(self.path, **self.kwargs)
            return self._data_df

    def _split(self, ratio, shuffle, nl_symbol):
        try:
            index = self.data_df.loc[
                ~self.data_df[self._cols[1]].str.contains(nl_symbol)].index
        except (AttributeError, TypeError, ValueError):
            index = self.data_df.loc[
                self.data_df[self._cols[1]].notna()].index
        index = index.to_list()
        if shuffle:
            random.shuffle(index)
        splitter = int(len(index)*ratio)
        self._training_rows = index[:splitter]
        self._testing_rows = index[splitter:]
        return self._training_rows, self._testing_rows

    def load_data(self, cols, ratio=0.9, shuffle=True, nl_symbol="_"):
        """ Load the training and testing sets.
        params:
        cols (list): ["input column name", "label column name"]
        ratio (float): the ratio between training and testing set.
        nl_symbol: the symbol denoting data with no label in the dataset.
        ================================================================
        return:
        data (list): [x_train, y_train, x_test, y_test]
        """
        self._cols = cols
        data_rows = self._split(ratio, shuffle, nl_symbol)
        data = list()
        for rows, col in itertools.product(data_rows, cols):
            data.append(self.data_df.loc[rows, col].to_numpy())
        return data

    def load_unlabeled(self, cols, nl_symbol="_"):
        """ Load the unlabeled data.
        params:
        cols (list): ["input column name", "label column name"]
        nl_symbol: the symbol denoting data with no label dataset.
        ================================================================
        return:
        data (numpy array): unlabeled dataset with respect to the specified
                            label column
        """
        try:
            index = self.data_df.loc[
                self.data_df[cols[-1]].str.contains(nl_symbol)].index
        except (AttributeError, TypeError, ValueError):
            index = self.data_df.loc[
                self.data_df[cols[-1]].isna()].index
        return self.data_df.loc[index, cols].to_numpy()

    def load_col(self, col_name):
        r""" Load a single column from the cvs file
        col_name (str): Name of the column.
        =======================================================================
        return (numpy array): Data in the colume.
        """
        return self.data_df[col_name].to_numpy()
