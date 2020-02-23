from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec
from ..utils.training_utils import open_log
from .training_args import ConventionalArgs


def init_data(data_path):
    data_loader = CVSLoader(data_path)
    x_train, y_train, x_test, y_test = data_loader.load_data(
        ["ECFP", "onehot_label"],
        ratio=0.7,
        shuffle=True
    )
    convert2vec_float = partial(convert2vec, dtype=float)
    x_train, y_train, x_test, y_test = list(
        map(convert2vec_float, [x_train, y_train, x_test, y_test]))
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)
    return x_train, y_train, x_test, y_test


def experiment(data_path, model, log_path):
    x_train, y_train, x_test, y_test = init_data(data_path)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    log_f, log_path = open_log(log_path)
    log_f.write("Experiment with random forest. Accuracy is: {}\n".format(acc))
    log_f.close()
    return acc, model


def experiment_rf(data_path, log_path, n_estimators, max_depth):
    r""" Test random forest
    """
    model = RandomForestClassifier(bootstrap=True,
                                   max_depth=max_depth,
                                   max_features=1024,
                                   n_estimators=n_estimators,
                                   n_jobs=-1)
    acc, model = experiment(data_path, model, log_path)
    return acc, model


def experiment_boost(data_path, log_path, n_estimators, max_depth):
    model = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators,
        learning_rate=0.5
    )
    acc, model = experiment(data_path, model, log_path)
    return acc, model


if __name__ == "__main__":
    import os

    parser = ConventionalArgs()
    args = parser.parse_args()
    experiment_rf(args.data_path,
                  os.path.join(args.log_path, "random_forest"),
                  args.n_estimators,
                  args.max_depth)

    experiment_boost(args.data_path,
                     os.path.join(args.log_path, "adaboost"),
                     args.n_estimators,
                     args.max_depth)
