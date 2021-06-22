###############################################################
# Train HMLC, HMLC_M, HMLC_L models with Noisy Student method #
###############################################################

from functools import partial

import tensorflow as tf
import numpy as np

from ...data_loaders.json_loader import JsonLoader
from ...utils.label_convertors import convert2vec, fill_unlabeled
from ..training_args import LMMixupArgs
from ..experiment_linear_exploit_partial import ExperimentLinearExploitPartial
from ...models.linear import Linear_S, Linear_M, Linear_L
from ...utils.training_utils import init_model, callback_list, training_log


class ExperimentLinearGinFP(ExperimentLinearExploitPartial):
    def load_data(self):
        data_loader = JsonLoader(self.data_path, rand_seed=self.rand_seed)
        x_train, y_train, x_test, y_test = data_loader.load_data(ratio=0.7)
        convert2vec_float = partial(convert2vec, dtype=float)
        x_train, y_train, x_test, y_test = list(
            map(convert2vec_float, [x_train, y_train, x_test, y_test])
        )

        if self.mixup is not None:
            x_train, y_train = self._mixup(x_train, y_train)
        x_unlabeled, y_partial = data_loader.load_unlabeled()

        return x_train, y_train, x_test, y_test, x_unlabeled, y_partial

    def train_teacher(
        self,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        x_pred,
        y_partial,
        log_f,
        log_path,
        n_repeat,
        activation="sigmoid",
        loss="binary_crossentropy",
        out_len=12,
    ):
        r"""Train linear model with Noisy Student and exploit the partially
            labeled data
        model: the model to be trained
        x_train: training data
        y_train: labels of the training data
        x_test: testing data
        y_test: testing data labels
        x_pred: unlabeled training data
        y_partial: the partial labels
        log_f: logging file handler
        log_path: path to the logging directory
        n_repeat: times to re-train the model with balanced data
        =======================================================================
        return: the trained model, training histories
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
            batch_size=self.batch_size,
            epochs=self.epochs,
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
            # label partially labled data
            y_pred = model.predict(x_pred)
            y_pred = fill_unlabeled(y_pred, y_partial, normalize=True)
            x_mix = np.concatenate([x_train, x_pred], axis=0)
            y_mix = np.concatenate([y_train, y_pred], axis=0)
            # shuffle
            randomed_idx = np.random.permutation(x_mix.shape[0])
            np.take(x_mix, randomed_idx, axis=0, out=x_mix)
            np.take(y_mix, randomed_idx, axis=0, out=y_mix)
            # mixup
            if self.mixup is not None:
                x_mix, y_mix = self._mixup(x_mix, y_mix,)
            # train model with the mixed data
            train_his = model.fit(
                x=x_mix,
                y=y_mix,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=cb_list,
                validation_data=[x_test, y_test],
            )
            histories.append(train_his)
            # log training history
            y_val = model.predict(x_test)
            training_log(train_his, y_val, y_test, log_f)

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
        y_partial,
        log_f,
        log_path,
        n_repeat,
        activation="sigmoid",
        loss="binary_crossentropy",
        out_len=12,
    ):
        r"""Train student linear model with Noisy Student and partially
            labeled data
        student_model: the model to be trained
        teacher_model: trained model used to generate labels
        x_train: training data
        y_train: labels of the training data
        x_test: testing data
        y_test: testing data labels
        x_pred: unlabeled training data
        y_partial: partial labels
        cb_list: callback list
        log_f: logging file handler
        log_path: path to the logging directory
        n_repeat: times to train the model
        ===========================================================================
        return: the trained model, training histories
        """
        # label partially labeld data with the teacher model
        y_pred = teacher_model.predict(x_pred)
        y_pred = fill_unlabeled(y_pred, y_partial, normalize=True)
        x_mix = np.concatenate([x_train, x_pred], axis=0)
        y_mix = np.concatenate([y_train, y_pred], axis=0)
        # shuffle
        randomed_idx = np.random.permutation(x_mix.shape[0])
        np.take(x_mix, randomed_idx, axis=0, out=x_mix)
        np.take(y_mix, randomed_idx, axis=0, out=y_mix)
        # mixup
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
        # fit Linear_M model to mixed dataset
        histories = list()
        log_f.write("training {}:\n".format(str(model)))
        train_his = model.fit(
            x=x_mix,
            y=y_mix,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=cb_list,
            validation_data=[x_test, y_test],
        )
        histories.append(train_his)

        y_val = model.predict(x_test)
        training_log(train_his, y_val, y_test, log_f)

        # repeat training the model
        for i in range(n_repeat):
            log_f.write(
                "repeat training {}, {}/{}:\n".format(str(model), i + 1, n_repeat)
            )
            # label partially labeld data
            y_pred = model.predict(x_pred)
            y_pred = fill_unlabeled(y_pred, y_partial, normalize=True)
            x_mix = np.concatenate([x_train, x_pred], axis=0)
            y_mix = np.concatenate([y_train, y_pred], axis=0)
            # shuffle
            randomed_idx = np.random.permutation(x_mix.shape[0])
            np.take(x_mix, randomed_idx, axis=0, out=x_mix)
            np.take(y_mix, randomed_idx, axis=0, out=y_mix)
            # mixup
            if self.mixup is not None:
                x_mix, y_mix = self._mixup(x_mix, y_mix)
            # train model with the mixed data
            train_his = model.fit(
                x=x_mix,
                y=y_mix,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=cb_list,
                validation_data=[x_test, y_test],
            )
            histories.append(train_his)
            # log training history
            y_val = model.predict(x_test)
            training_log(train_his, y_val, y_test, log_f)

        return model, histories

    def run_experiment(self):
        # load training and testing data
        x_train, y_train, x_test, y_test, x_unlabeled, y_partial = self.load_data()
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
            y_partial=y_partial,
            log_f=log_f,
            log_path=log_path,
            n_repeat=self.n_repeat,
            activation="sigmoid",
            loss="binary_crossentropy",
            out_len=12,
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
                y_partial=y_partial,
                log_f=log_f,
                log_path=log_path,
                n_repeat=self.n_repeat,
                activation="sigmoid",
                loss="binary_crossentropy",
                out_len=12,
            )
            # log results
            self.log_training(trained_model, histories, log_path)

        log_f.write("best losses:\n {}\n".format(str(self.best_loss)))
        log_f.write("best accuracies:\n {}\n".format(str(self.best_acc)))
        log_f.close()

        self.log_predictions(trained_model, x_test, y_test, log_path)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    parser = LMMixupArgs()
    args = parser.parse_args()
    experiment = ExperimentLinearGinFP(
        data_path=args.data_path,
        log_path=args.log_path,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_repeat=args.repeat,
        mixup=args.mixup,
        mixup_repeat=args.mixup_repeat,
        learning_rate=args.learning_rate,
        rand_seed=args.rand_seed,
    )
    experiment.run_experiment()
