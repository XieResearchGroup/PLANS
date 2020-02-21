from functools import partial

import numpy as np

from ..models.linear import Linear_S, Linear_M, Linear_L
from .train_model import ns_linear_teacher_model, ns_linear_student_model
from .train_model import predict_and_mix, plot_history
from .training_args import LMOutsideDataArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec
from ..utils.training_utils import init_model, callback_list, open_log


def main(data_path,
         outside_data_path,
         log_path,
         es_patience,
         batch_size,
         epochs,
         n_repeat):
    # data
    data_loader = CVSLoader(data_path)
    outside_data_loader = CVSLoader(outside_data_path)
    x_train, y_train, x_test, y_test = data_loader.load_data(
        ["ECFP", "onehot_label"],
        ratio=0.7,
        shuffle=True)
    convert2vec_float = partial(convert2vec, dtype=float)
    x_train, y_train, x_test, y_test = list(
        map(convert2vec_float, [x_train, y_train, x_test, y_test]))
    x_cyp_unlabeled = data_loader.load_unlabeled(["ECFP", "onehot_label"])
    x_cyp_unlabeled = convert2vec(x_cyp_unlabeled[:, 0], dtype=float)
    x_outside = outside_data_loader.load_col("ECFP")
    x_outside = convert2vec(x_outside, dtype=float)
    x_unlabeled = np.concatenate([x_cyp_unlabeled, x_outside], axis=0)

    # Open log
    log_f, log_path = open_log(log_path)

    # init model Linear_S
    model = init_model(Linear_S)
    # callbacks
    cb_list = callback_list(log_path, es_patience, model)
    # fit
    trained_model, histories = ns_linear_teacher_model(
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
        log_path=log_path,
        n_repeat=n_repeat
    )
    # plot the training history
    plot_history(histories, log_path, str(trained_model))

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
    trained_model, histories = ns_linear_student_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_mix=x_mix,
        y_mix=y_mix,
        x_test=x_test,
        y_test=y_test,
        x_pred=x_unlabeled,
        batch_size=batch_size,
        epochs=epochs,
        cb_list=cb_list,
        log_f=log_f,
        log_path=log_path,
        n_repeat=n_repeat
    )
    # plot the training history
    plot_history(histories, log_path, str(trained_model))

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
    trained_model, histories = ns_linear_student_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_mix=x_mix,
        y_mix=y_mix,
        x_test=x_test,
        y_test=y_test,
        x_pred=x_unlabeled,
        batch_size=batch_size,
        epochs=epochs,
        cb_list=cb_list,
        log_f=log_f,
        log_path=log_path,
        n_repeat=n_repeat
    )
    # plot the training history
    plot_history(histories, log_path, str(trained_model))

    log_f.close()


if __name__ == "__main__":
    parser = LMOutsideDataArgs()
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        outside_data_path=args.outside_path,
        log_path=args.log_path,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_repeat=args.repeat
    )
