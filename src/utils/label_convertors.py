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
    l1 = _if_true_label(true_label)
    l2_1 = _if_true_label(true_label[:, 0:2])
    l2_2 = _if_true_label(
        true_label[:, 2:4])
    l2_3 = _if_true_label(true_label[:, 4])
    return np.concatenate(
        [l1, l2_1, l2_2, l2_3, true_label, true_label], axis=1)


def convert2hier(label, dtype):
    label = convert2vec(label, dtype=int)
    label = hierarchical(label)
    return label.astype(dtype)