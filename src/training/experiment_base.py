from functools import partial

from ..models.linear import Linear_S, Linear_M, Linear_L
from .train_model import predict_and_mix, plot_history
from .training_args import LMMixupArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec
from ..utils.mixup import mixup
from ..utils.training_utils import init_model, callback_list, open_log
from ..utils.training_utils import find_best
from ..utils.training_utils import training_log


class ExperimentBase:
    def __init__(
        self,
        data_path,
        log_path,
        es_patience,
        batch_size,
        epochs,
        n_repeat,
        rand_seed,
        mixup=None,
        mixup_repeat=None,
        learning_rate=1e-6,
        drop_rate=0.3,
    ):
        self.data_path = data_path
        self.log_path = log_path
        self.es_patience = es_patience
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_repeat = n_repeat
        self.rand_seed = rand_seed
        self.mixup = mixup
        self.mixup_repeat = mixup_repeat
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.best_loss = dict()
        self.best_acc = dict()

    def _mixup(self, x, y):
        x_mix, y_mix = mixup(self.mixup, self.mixup, x, y, repeat=self.mixup_repeat)
        return x_mix, y_mix

    def load_data(self, cols=["ECFP", "onehot_label"], ratio=0.7, shuffle=True):
        data_loader = CVSLoader(self.data_path, rand_seed=self.rand_seed)
        x_train, y_train, x_test, y_test = data_loader.load_data(
            cols, ratio=ratio, shuffle=shuffle
        )
        convert2vec_float = partial(convert2vec, dtype=float)
        x_train, y_train, x_test, y_test = list(
            map(convert2vec_float, [x_train, y_train, x_test, y_test])
        )
        if self.mixup is not None:
            x_train, y_train = self._mixup(x_train, y_train)
        x_unlabeled = data_loader.load_unlabeled(cols)
        x_unlabeled = convert2vec(x_unlabeled[:, 0], dtype=float)
        return x_train, y_train, x_test, y_test, x_unlabeled

    def open_log_(self, log_path):
        return open_log(log_path)

    def train_teacher(
        self,
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
        n_repeat,
        activation="softmax",
        loss="categorical_crossentropy",
        out_len=32,
    ):
        r""" Train linear model with Noisy Student
        model: the model to be trained
        x_train: training data
        y_train: labels of the training data
        x_test: testing data
        y_test: testing data labels
        x_pred: unlabeled training data
        batch_size: size of mini batches
        epochs: number of epochs
        log_f: logging file handler
        log_path: path to the logging directory
        n_repeat: times to train the model
        =======================================================================
        return: the trained model, training history
        """
        model = init_model(
            model,
            drop_rate=self.drop_rate,
            loss=loss,
            out_len=out_len,
            activation=activation,
        )
        cb_list = callback_list(
            log_path, self.es_patience, model, learning_rate=self.learning_rate
        )
        histories = list()
        log_f.write("training {}:\n".format(str(model)))
        train_his = model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=cb_list,
            validation_data=[x_test, y_test],
        )
        histories.append(train_his)

        y_pred = model.predict(x_test)
        training_log(train_his, y_pred, y_test, log_f)

        # repeat training the model
        for i in range(n_repeat):
            log_f.write(
                "repeat training {}, {}/{}:\n".format(str(model), i + 1, n_repeat)
            )
            # label unlabled
            x_mix, y_mix = predict_and_mix(
                model=model, x_pred=x_pred, x=x_train, y=y_train, shuffle=True
            )
            if self.mixup is not None:
                x_mix, y_mix = self._mixup(x_mix, y_mix)
            # train model with the mixed data
            train_his = model.fit(
                x=x_mix,
                y=y_mix,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=cb_list,
                validation_data=[x_test, y_test],
            )
            histories.append(train_his)
            # log training history
            y_pred = model.predict(x_test)
            training_log(train_his, y_pred, y_test, log_f)

        return model, histories

    def train_student(
        self,
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
        n_repeat,
        activation="softmax",
        loss="categorical_crossentropy",
        out_len=32,
    ):
        r""" Train student linear model with Noisy Student
        student_model: the model to be trained
        teacher_model: the trained model for generating the first batch of
                       predictions
        x_train: training data
        y_train: labels of the training data
        x_test: testing data
        y_test: testing data label
        x_pred: unlabeled training data
        batch_size: size of mini batches
        epochs: number of epochs
        log_f: logging file handler
        log_path: path to the logging directory
        n_repeat: times to train the model
        =======================================================================
        return: the trained model, training history
        """
        x_mix, y_mix = predict_and_mix(
            teacher_model, x_pred, x_train, y_train, shuffle=True
        )
        if self.mixup is not None:
            x_mix, y_mix = self._mixup(x_mix, y_mix)
        # init model
        model = init_model(
            student_model,
            drop_rate=self.drop_rate,
            loss=loss,
            out_len=out_len,
            activation=activation,
        )
        # callbacks
        cb_list = callback_list(
            log_path, self.es_patience, model, learning_rate=self.learning_rate
        )
        histories = list()
        log_f.write("training {}:\n".format(str(model)))
        train_his = model.fit(
            x=x_mix,
            y=y_mix,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=cb_list,
            validation_data=[x_test, y_test],
        )
        histories.append(train_his)

        y_pred = model.predict(x_test)
        training_log(train_his, y_pred, y_test, log_f)

        # repeat training the model
        for i in range(n_repeat):
            log_f.write(
                "repeat training {}, {}/{}:\n".format(str(model), i + 1, n_repeat)
            )
            # label unlabled
            x_mix, y_mix = predict_and_mix(
                model=model, x_pred=x_pred, x=x_train, y=y_train, shuffle=True
            )
            if self.mixup is not None:
                x_mix, y_mix = self._mixup(x_mix, y_mix)
            # train model with the mixed data
            train_his = model.fit(
                x=x_mix,
                y=y_mix,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=cb_list,
                validation_data=[x_test, y_test],
            )
            histories.append(train_his)
            # log training history
            y_pred = model.predict(x_test)
            training_log(train_his, y_pred, y_test, log_f)

        return model, histories

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
        x_train, y_train, x_test, y_test, x_unlabeled = self.load_data()
        # open log
        log_f, log_path = self.open_log_(self.log_path)
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
            n_repeat=self.n_repeat,
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
                n_repeat=self.n_repeat,
            )
            # log results
            self.log_training(trained_model, histories, log_path)

        log_f.write("best losses:\n {}\n".format(str(self.best_loss)))
        log_f.write("best accuracies:\n {}\n".format(str(self.best_acc)))
        log_f.close()


if __name__ == "__main__":
    parser = LMMixupArgs()
    args = parser.parse_args()
    experiment = ExperimentBase(
        data_path=args.data_path,
        log_path=args.log_path,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_repeat=args.repeat,
        mixup=args.mixup,
        mixup_repeat=args.mixup_repeat,
        rand_seed=args.rand_seed,
    )
    experiment.run_experiment()
