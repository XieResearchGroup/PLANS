from functools import partial

from ..models.linear import Linear_S, Linear_M, Linear_L
from .train_model import ns_linear_model, predict_and_mix
from .training_args import TrainingArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec
from ..utils.training_utils import init_model, callback_list
from ..utils.training_utils import open_log


def main(data_path, log_path, es_patience, batch_size, epochs):
    # data
    data_loader = CVSLoader(data_path)
    x_train, y_train, x_test, y_test = data_loader.load_data(
        ["ECFP", "onehot_label"],
        ratio=0.7,
        shuffle=True)
    convert2vec_float = partial(convert2vec, dtype=float)
    x_train, y_train, x_test, y_test = list(
        map(convert2vec_float, [x_train, y_train, x_test, y_test]))
    x_unlabeled = data_loader.load_unlabeled(["ECFP", "onehot_label"])
    x_unlabeled = convert2vec(x_unlabeled[:, 0], dtype=float)

    # Open log
    log_f = open_log(log_path)

    # init model Linear_S
    model = init_model(Linear_S)
    # callbacks
    cb_list = callback_list(log_path, es_patience, model)
    # fit
    trained_model = ns_linear_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_pred=x_unlabeled,
        batch_size=batch_size,
        epochs=epochs,
        cb_list=cb_list,
        log_f=log_f,
        n_repeat=3
    )
    x_mix, y_mix = predict_and_mix(
        trained_model,
        x_unlabeled,
        x_train,
        y_train,
        shuffle=True
    )

    # init model Linear_M
    model = init_model(Linear_M)
    # callbacks
    cb_list = callback_list(log_path, es_patience, model)
    # fit Linear_M model to mixed dataset
    trained_model = ns_linear_model(
        model=model,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_test=y_test,
        x_pred=x_unlabeled,
        batch_size=batch_size,
        epochs=epochs,
        cb_list=cb_list,
        log_f=log_f,
        n_repeat=3
    )
    x_mix, y_mix = predict_and_mix(
        trained_model,
        x_unlabeled,
        x_train,
        y_train,
        shuffle=True
    )

    # init model Linear_L
    model = init_model(Linear_L)
    # callbacks
    cb_list = callback_list(log_path, es_patience, model)
    # fit Linear_L model to mixed dataset
    ns_linear_model(
        model=model,
        x_train=x_mix,
        y_train=y_mix,
        x_test=x_test,
        y_test=y_test,
        x_pred=x_unlabeled,
        batch_size=batch_size,
        epochs=epochs,
        cb_list=cb_list,
        log_f=log_f,
        n_repeat=3
    )

    log_f.close()


if __name__ == "__main__":
    parser = TrainingArgs()
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        log_path=args.log_path,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
