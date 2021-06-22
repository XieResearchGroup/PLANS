###########################################################
# Helper functions for mixup                              #
###########################################################

import numpy as np


def _reshape_label(label):
    if len(label.shape) == 1:
        return label.reshape(-1, 1)
    return label


def _mixup_samples(alpha, beta, samp1, label1, samp2, label2, repeat, shuffle=True):
    """ mixup two samples
    alpha, beta: parameters of Beta distribution
    samp1, samp2: samples to be mixed
    label1, label2: labels of the samples to be mixed
    repeat (int): times to repeat the inputs
    shuffle (bool): whether shuffle the inputs
    """
    assert samp1.shape == samp2.shape
    assert label1.shape == label2.shape

    # reshape labels if neccesary
    label1 = _reshape_label(label1)
    label2 = _reshape_label(label2)

    # repeat
    samp1 = np.repeat(samp1, repeat, axis=0)
    label1 = np.repeat(label1, repeat, axis=0)
    samp2 = np.repeat(samp2, repeat, axis=0)
    label2 = np.repeat(label2, repeat, axis=0)

    # shuffle inputs
    if shuffle:
        indices_1 = np.random.permutation(samp1.shape[0])
        indices_2 = np.random.permutation(samp1.shape[0])
        samp1 = np.take(samp1, indices_1, axis=0)
        label1 = np.take(label1, indices_1, axis=0)
        samp2 = np.take(samp2, indices_2, axis=0)
        label2 = np.take(label2, indices_2, axis=0)

    # sample lambda from beta distribution
    lamb = np.random.beta(alpha, beta, (samp1.shape[0], 1))

    # mixup samples
    sample = lamb * samp1 + (1 - lamb) * samp2
    label = lamb * label1 + (1 - lamb) * label2

    return sample, label


def mixup(
    alpha,
    beta,
    sample_1,
    label_1,
    *,
    sample_2=None,
    label_2=None,
    repeat=1,
    shuffle=True
):
    """ The interface of mixup functions.
    """
    if sample_2 is None:
        return _mixup_samples(
            alpha, beta, sample_1, label_1, sample_1, label_1, repeat, shuffle
        )
    else:
        return _mixup_samples(
            alpha, beta, sample_1, label_1, sample_2, label_2, repeat, shuffle
        )
