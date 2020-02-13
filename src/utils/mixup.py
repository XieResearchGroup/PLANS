###########################################################
# Helper functions for mixup                              #
###########################################################

import numpy as np


def _reshape_label(label):
    if len(label.shape) == 1:
        return label.reshape(-1, 1)
    return label


def _mixup_one_sample(alpha, beta, samples, labels):
    """ mixup one group of samples by random permutation
    """
    # create another group of samples by randomly permulate the samples
    random_indices = np.random.permutation(samples.shape[0])
    samp2 = np.take(samples, random_indices, axis=0)
    label2 = np.take(labels, random_indices, axis=0)

    return _mixup_two_samples(alpha, beta, samples, labels, samp2, label2)


def _mixup_two_samples(alpha, beta, samp1, label1, samp2, label2):
    """ mixup two samples
    alpha, beta: parameters of Beta distribution
    samp1, samp2: samples to be mixed
    label1, label2: labels of the samples to be mixed
    """
    # find out the shape of the samples
    assert samp1.shape == samp2.shape
    sam_shape = (samp1.shape[0], 1)

    # reshape labels if neccesary
    assert label1.shape == label2.shape
    label1 = _reshape_label(label1)
    label2 = _reshape_label(label2)

    # sample lambda from beta distribution
    lamb = np.random.beta(alpha, beta, sam_shape)

    # mixup samples
    sample = lamb * samp1 + (1-lamb) * samp2
    label = lamb * label1 + (1-lamb) * label2

    return sample, label


def mixup(alpha, beta, sample_1, label_1, sample_2=None, label_2=None):
    """ The interface of mixup functions.
    """
    if sample_2 is None:
        return _mixup_one_sample(alpha, beta, sample_1, label_1)
    else:
        return _mixup_two_samples(
            alpha, beta, sample_1, label_1, sample_2, label_2)
