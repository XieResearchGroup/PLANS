import numpy as np

def _generate_weight(label, nl_symbol, n_fold):
    """ Generate weights based on one multi-label label.
    label (str): multi-label or multi-class label
    nl_symbol (str): the symbol representing no label
    n_fold (float): unlabeled has n_fold weight value comparing to labeled
    ===========================================================================
    return (list): weights
    """
    weights = [1] * len(label)
    for i, lbl in enumerate(label):
        if lbl == nl_symbol:
            weights[i] = n_fold
    return weights


def generate_partially_unlabeled_weights(labels, nl_symbol="_", n_fold=0.5):
    """ Generate training loss weights for the patial labels
    labels (1d numpy.array or list of str): partial labels with nl_symbols
    nl_symbol (str): symbol used in the labels to denote no label
    ===========================================================================
    return (numpy.array): label weights with unlabeled data have n_fold folds
        weights comparing to labeled data
    """
    weights = np.zeros((len(labels), len(labels[0])))
    for i, label in enumerate(labels):
        weight = _generate_weight(label, nl_symbol, n_fold)
        weights[i] = weight
    return weights


