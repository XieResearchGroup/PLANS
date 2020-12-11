from functools import partial

from ..models.linear import Linear_S, Linear_M, Linear_L
from .train_model import ns_linear_teacher_model, ns_linear_student_model
from .train_model import predict_and_mix, plot_history
from .training_args import LinearModelTrainingArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec
from ..utils.training_utils import init_model, callback_list, open_log
from ..utils.training_utils import find_best


def main(data_path,
         log_path,
         es_patience,
         batch_size,
         epochs,
         n_repeat,
         rand_seed=None):
    # data
    data_loader = CVSLoader(data_path, rand_seed=rand_seed)
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
    log_f, log_path = open_log(log_path)
    # dicts for evaluating training results
    best_loss = dict()
    best_acc = dict()

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
    # log best results
    min_loss = find_best(histories, "val_loss", "min")
    max_acc = find_best(histories, "val_acc", "max")
    best_loss[str(trained_model)] = min_loss
    best_acc[str(trained_model)] = max_acc

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
    # log best results
    min_loss = find_best(histories, "val_loss", "min")
    max_acc = find_best(histories, "val_acc", "max")
    best_loss[str(trained_model)] = min_loss
    best_acc[str(trained_model)] = max_acc

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
    # log best results
    min_loss = find_best(histories, "val_loss", "min")
    max_acc = find_best(histories, "val_acc", "max")
    best_loss[str(trained_model)] = min_loss
    best_acc[str(trained_model)] = max_acc

    log_f.write("best losses\n {}\n".format(str(best_loss)))
    log_f.write("best accuracies\n {}\n".format(str(best_acc)))

    log_f.close()


if __name__ == "__main__":
    parser = LinearModelTrainingArgs()
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        log_path=args.log_path,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_repeat=args.repeat,
        rand_seed=args.rand_seed
    )
