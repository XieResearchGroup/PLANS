import os
from functools import partial

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec
from ..utils.training_utils import open_log
from .training_args import ConventionalArgs


def init_data(data_path, rand_seed):
    r""" Load data and convert labels from onehot to integers.
    """
    data_loader = CVSLoader(data_path, rand_seed=rand_seed)
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


def experiment(data_path, model, log_path, rand_seed):
    r""" Run experiment
    """
    x_train, y_train, x_test, y_test = init_data(data_path, rand_seed)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_pred, y_test)
    # Logging the experiment results
    log_f, log_path = open_log(log_path)
    log_f.write(
        "Experiment with {}. Accuracy is: {}\n".format(
            type(model).__name__, acc))
    # Write prediction and true label
    log_f.write("@prediction-truth\n")
    for p, t in zip(y_pred, y_test):
        log_f.write(str(p)+" "+str(t)+"\n")
    log_f.write("="*80+"\n")
    log_f.close()
    return acc, model


def experiment_rf(data_path,
                  log_path,
                  n_estimators,
                  max_depth,
                  rand_seed=None):
    r""" Test random forest
    """
    model = RandomForestClassifier(bootstrap=True,
                                   max_depth=max_depth,
                                   max_features=1024,
                                   n_estimators=n_estimators,
                                   n_jobs=-1)
    acc, model = experiment(data_path, model, log_path, rand_seed)
    return acc, model


def experiment_boost(data_path,
                     log_path,
                     n_estimators,
                     max_depth,
                     rand_seed=None):
    model = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=max_depth),
        n_estimators=n_estimators,
        learning_rate=0.5
    )
    acc, model = experiment(data_path, model, log_path, rand_seed)
    return acc, model


def experiment_svm(data_path, log_path, rand_seed=None):
    model = SVC(C=1.0, gamma="scale")
    acc, model = experiment(data_path, model, log_path, rand_seed)
    return acc, model


def experiment_xgboost(data_path,
                       log_path,
                       max_depth,
                       num_class=32,
                       n_round=2,
                       rand_seed=None):
    import xgboost as xgb

    # init data
    x_train, y_train, x_test, y_test = init_data(data_path, rand_seed)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)
    # setup parameters
    param = {}
    param['objective'] = 'multi:softmax'
    param['eta'] = 0.1
    param['max_depth'] = max_depth
    param['silent'] = 1
    param['nthread'] = int(os.cpu_count()/2)
    param['num_class'] = num_class
    bst = xgb.train(param, dtrain, n_round)
    preds = bst.predict(dtest).astype(int)
    acc = accuracy_score(preds, y_test)
    # Logging the experiment results
    log_f, log_path = open_log(log_path)
    log_f.write(
        "Experiment with xgboost. Accuracy is: {}\n".format(acc))
    # Write prediction and true label
    log_f.write("@prediction-truth\n")
    for p, t in zip(preds, y_test):
        log_f.write(str(p)+" "+str(t)+"\n")
    log_f.write("="*80+"\n")
    log_f.close()
    return acc, bst


if __name__ == "__main__":
    parser = ConventionalArgs()
    args = parser.parse_args()
    experiment_rf(args.data_path,
                  os.path.join(args.log_path, "random_forest"),
                  args.n_estimators,
                  args.max_depth,
                  args.rand_seed)

    experiment_boost(args.data_path,
                     os.path.join(args.log_path, "adaboost"),
                     args.n_estimators,
                     args.max_depth,
                     args.rand_seed)

    experiment_svm(args.data_path,
                   os.path.join(args.log_path, "svm"),
                   args.rand_seed)

    experiment_xgboost(args.data_path,
                       os.path.join(args.log_path, "xgboost"),
                       args.max_depth,
                       rand_seed=args.rand_seed)
