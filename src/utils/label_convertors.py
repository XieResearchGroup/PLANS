import numpy as np


def convert2vec(data, dtype=int):
    data = data.tolist()
    data = list(map(list, data))
    data = [list(map(dtype, d)) for d in data]
    data = np.array(data)
    return data


def _if_true_label(label):
    try:
        return np.expand_dims((np.sum(label, axis=1)>0).astype(np.int32), 1)
    except np.AxisError:
        return np.expand_dims(((label>0).astype(np.int32)), 1)


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
    data_labeled = np.zeros(predictions.shape)
    for i, data in enumerate(data_unlabeled):
        labeled = list(data)
        for j, label in enumerate(labeled):
            try:
                labeled[j] = int(label)
            except ValueError:
                if hard_label:
                    labeled[j] = round(predictions[i, j])
                else:
                    labeled[j] = predictions[i, j]
        data_labeled[i] = labeled
    return data_labeled