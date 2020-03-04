from itertools import product

import numpy as np


def convert2vec(data, dtype=int):
    data = data.tolist()
    data = list(map(list, data))
    data = [list(map(dtype, d)) for d in data]
    data = np.array(data)
    return data


def _if_true_label(label):
    try:
        return np.expand_dims((np.sum(label, axis=1) > 0).astype(np.int32), 1)
    except np.AxisError:
        return np.expand_dims(((label > 0).astype(np.int32)), 1)


def hierarchical(true_label):
    rounded = np.round(true_label)
    l1 = _if_true_label(rounded)
    l2_1 = _if_true_label(rounded[:, 0:2])
    l2_2 = _if_true_label(
        rounded[:, 2:4])
    l2_3 = _if_true_label(rounded[:, 4])
    return np.concatenate(
        [l1, l2_1, l2_2, l2_3, rounded, true_label], axis=1)


def convert2hier(label, dtype):
    label = convert2vec(label, dtype=int)
    label = hierarchical(label)
    return label.astype(dtype)


def fill_unlabeled(predictions, data_unlabeled, hard_label=False):
    """ Fill the unlabeled blanks in data_unlabeled with predicted labels
    predictions (numpy.array): predicted labels, shape is (?, 5)
    data_unlabeled (numpy.array): str, unlabeled data in "1_10_"-like format
    hard_label (bool): use hard label to label the unlabeled data
    ========================================================================
    return: numpy.array
    """
    data_labeled = np.zeros((len(data_unlabeled), len(data_unlabeled[0])))
    for i, data in enumerate(data_unlabeled):
        labeled = list(data)
        for j, label in enumerate(labeled):
            try:
                labeled[j] = int(label)
            except ValueError:
                if hard_label:
                    if isinstance(predictions, (int, float)):
                        labeled[j] = round(predictions)
                    else:
                        labeled[j] = round(predictions[i, j])
                else:
                    if isinstance(predictions, (int, float)):
                        labeled[j] = predictions
                    else:
                        labeled[j] = predictions[i, j]
        data_labeled[i] = labeled
    return data_labeled


def multilabel2onehot(multilabel: str):
    """ Convert multilabel to onehot
    multilabel (str): a multi-label with format like "10010"
    ========================================================
    return (str): onehot label as a string
    """
    # decide the length of the one-hot label
    length = 2 ** len(multilabel)
    onehot = [0] * length
    # get the one-hot vector
    onehot[int(multilabel, 2)] = 1
    return "".join(map(str, onehot))


def vec2onehot(vec):
    r""" Convert a multilabel vector to one hot
    """
    label = "".join(list(map(str, map(int, vec))))
    onehot = [0] * (2 ** len(label))
    onehot[int(label, 2)] = 1
    return onehot


def multivec2onehot(multilabel: np.array):
    r"""Convert multilabel to onehot
    multilabel (np.array): a multi-label numpy array
    ========================================================
    return (np.array): onehot label as numpy array
    """
    onehot = np.apply_along_axis(vec2onehot, 1, multilabel)
    return onehot


def onehot2vec(onehot):
    r""" Convert a onehot to multilabel vector
    """
    # get the positive class index
    index = np.argmax(onehot)
    # decide the output length
    ml_len = len(bin(len(onehot)-1)) - 2
    # get the str representation of the binarized index
    bin_index_str = bin(index)[2:]  # remove "0b"
    # construct the multilabel string
    vec_str = (ml_len - len(bin_index_str)) * "0" + bin_index_str
    # convert string to np.array
    vec = np.array(list(map(int, list(vec_str))))
    return vec


def onehot2multivec(onehotlabel: np.array):
    r"""Convert onehot to multilabel
    onehotlabel (np.array): a onehot-label numpy array
    ========================================================
    return (np.array): multilabel as numpy array
    """
    multivec = np.apply_along_axis(onehot2vec, 1, onehotlabel)
    return multivec


def _fill_missing(label: str, digits: tuple) -> str:
    r""" Helper that fill the label with digits
    """
    i, j = 0, 0
    label = list(label)
    while j < len(digits):
        digit = digits[j]
        while label[i] != "_":
            i += 1
        label[i] = str(digit)
        j += 1
    return "".join(label)


def partial2onehot(label):
    r""" Convert a partial multi label to its potential one hot label. e.g.
        "10_" -> "100" or "101" -> "0000__00"
    label (str): a multilabel with or without missing label(s)
    ===========================================================================
    return (str): the corresponding potential one-hot label
    """
    # count number of missing lables
    n_miss = label.count("_")
    # decide the one-hot lable lenth
    len_oh = 2 ** len(label)
    oh = ["0"] * len_oh
    for i in product([0, 1], repeat=n_miss):
        binary = _fill_missing(label, i)
        index = int(binary, 2)
        oh[index] = "_"
    return "".join(oh)
