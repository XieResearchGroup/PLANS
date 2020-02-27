from functools import partial

from ..models.linear import Linear_S, Linear_M, Linear_L
from .train_model import ns_linear_teacher_model, ns_linear_student_model
from .train_model import predict_and_mix, plot_history
from .training_args import LMMixupArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec
from ..utils.mixup import mixup
from ..utils.training_utils import init_model, callback_list, open_log
from ..utils.training_utils import find_best


class ExperimentLinearMixup():

    def __init__(self,
                 data_path,
                 log_path,
                 es_patience,
                 batch_size,
                 epochs,
                 n_repeat,
                 mixup,
                 mixup_repeat,
                 rand_seed=None):
        self.data_path = data_path
        self.log_path = log_path
        self.es_patience = es_patience
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_repeat = n_repeat
        self.mixup = mixup
        self.mixup_repeat = mixup_repeat
        self.rand_seed = rand_seed
        self.best_loss = dict()
        self.best_acc = dict()

    def _mixup(self, x, y):
        x_mix, y_mix = mixup(
            self.mixup, self.mixup, x, y, repeat=self.mixup_repeat)
        return x_mix, y_mix

    def load_data(self, data_path):
        data_loader = CVSLoader(data_path, rand_seed=self.rand_seed)
        x_train, y_train, x_test, y_test = data_loader.load_data(
            ["ECFP", "onehot_label"],
            ratio=0.7,
            shuffle=True)
        convert2vec_float = partial(convert2vec, dtype=float)
        x_train, y_train, x_test, y_test = list(
            map(convert2vec_float, [x_train, y_train, x_test, y_test]))
        x_train, y_train = self._mixup(x_train, y_train)
        x_unlabeled = data_loader.load_unlabeled(["ECFP", "onehot_label"])
        x_unlabeled = convert2vec(x_unlabeled[:, 0], dtype=float)
        return x_train, y_train, x_test, y_test, x_unlabeled

    def _open_log(self, log_path):
        return open_log(log_path)

    def train_teacher(self,
                      model,
                      x_train,
                      y_train,
                      x_test,
                      y_test,
                      x_pred,
                      batch_size,
                      epochs,
                      log_f,
                      log_path,
                      n_repeat):
        model = init_model(model)
        cb_list = callback_list(log_path, self.es_patience, model)
        trained_model, histories = ns_linear_teacher_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_pred=x_pred,
            batch_size=batch_size,
            epochs=epochs,
            cb_list=cb_list,
            log_f=log_f,
            log_path=log_path,
            n_repeat=n_repeat
        )
        return trained_model, histories

    def train_student(self,
                      student_model,
                      teacher_model,
                      x_train,
                      y_train,
                      x_test,
                      y_test,
                      x_pred,
                      batch_size,
                      epochs,
                      log_f,
                      log_path,
                      n_repeat):
        x_mix, y_mix = predict_and_mix(teacher_model,
                                       x_pred,
                                       x_train,
                                       y_train,
                                       shuffle=True)
        x_mix, y_mix = self._mixup(x_mix, y_mix)
        # init model
        model = init_model(student_model)
        # callbacks
        cb_list = callback_list(log_path, self.es_patience, model)
        # fit Linear_M model to mixed dataset
        trained_model, histories = ns_linear_student_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_mix=x_mix,
            y_mix=y_mix,
            x_test=x_test,
            y_test=y_test,
            x_pred=x_pred,
            batch_size=batch_size,
            epochs=epochs,
            cb_list=cb_list,
            log_f=log_f,
            log_path=log_path,
            n_repeat=n_repeat
        )
        return trained_model, histories

    def log_training(self, trained_model, histories, log_path):
        # plot the training history
        plot_history(histories, log_path, str(trained_model))
        # log best results
        min_loss = find_best(histories, "val_loss", "min")
        max_acc = find_best(histories, "val_acc", "max")
        self.best_loss[str(trained_model)] = min_loss
        self.best_acc[str(trained_model)] = max_acc

    def run_experiment(self):
        # load training and testing data
        x_train, y_train, x_test, y_test, x_unlabeled = self.load_data(
            self.data_path)
        # open log
        log_f, log_path = self._open_log(self.log_path)
        # train the teacher model
        trained_model, histories = self.train_teacher(
            model=Linear_S,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_pred=x_unlabeled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            log_f=log_f,
            log_path=log_path,
            n_repeat=self.n_repeat
        )
        # log results
        self.log_training(trained_model, histories, log_path)
        # train student models
        for student in [Linear_M, Linear_L]:
            trained_model, histories = self.train_student(
                student_model=student,
                teacher_model=trained_model,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                x_pred=x_unlabeled,
                batch_size=self.batch_size,
                epochs=self.epochs,
                log_f=log_f,
                log_path=log_path,
                n_repeat=self.n_repeat
            )
            # log results
            self.log_training(trained_model, histories, log_path)

        log_f.write("best losses:\n {}\n".format(str(self.best_loss)))
        log_f.write("best accuracies:\n {}\n".format(str(self.best_acc)))


if __name__ == "__main__":
    parser = LMMixupArgs()
    args = parser.parse_args()
    experiment = ExperimentLinearMixup(
        data_path=args.data_path,
        log_path=args.log_path,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_repeat=args.repeat,
        mixup=args.mixup,
        mixup_repeat=args.mixup_repeat,
        rand_seed=args.rand_seed
    )
    experiment.run_experiment()
