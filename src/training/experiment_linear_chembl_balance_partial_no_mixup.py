from functools import partial

import numpy as np
from tqdm import tqdm

from .experiment_linear_balance_classes import ExperimentLinearBalanced
from .experiment_linear_exploit_partial import ExperimentLinearExploitPartial
from ..models.linear import Linear_S, Linear_M, Linear_L
from .training_args import LMMixupOutsideDataArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..data_loaders.hdf5_loader import HDF5Loader
from ..utils.label_convertors import convert2vec
from ..utils.label_convertors import partial2onehot, fill_unlabeled
from ..utils.training_utils import init_model, callback_list, training_log


class ExpLinBalLargeOutsideExploitPartialNoMixup(
    ExperimentLinearBalanced, ExperimentLinearExploitPartial
):
    def load_data(self):
        data_loader = CVSLoader(self.data_path, rand_seed=self.rand_seed)
        outside_data_loader = HDF5Loader(self.outside_data_path, "r")
        x_train, y_train, x_test, y_test = data_loader.load_data(
            ["ECFP", "onehot_label"], ratio=0.7, shuffle=True
        )
        convert2vec_float = partial(convert2vec, dtype=float)
        x_train, y_train, x_test, y_test = list(
            map(convert2vec_float, [x_train, y_train, x_test, y_test])
        )
        # if self.mixup is not None:
        #     x_train, y_train = self._mixup(x_train, y_train)
        data_partial = data_loader.load_unlabeled(["ECFP", "Label"])
        x_partial = convert2vec(data_partial[:, 0], dtype=float)
        y_partial = data_partial[:, 1]
        for i, label in enumerate(y_partial):
            y_partial[i] = partial2onehot(label)
        return (
            x_train,
            y_train,
            x_test,
            y_test,
            x_partial,
            y_partial,
            outside_data_loader,
        )

    def _predict_and_balance(self, model, outside, x, y, shuffle=True):
        r""" Make predictions with the given model and mix it with the existing
        training set.
        model (tf.keras.Model): the pretrained model to make predictions
        x_unlabeled (np.array): inputs for the model
        x (np.arrapy): training set data
        y (np.array): training set label
        shuffle (bool): if shuffle after mixing the training and predicted data
        =======================================================================
        return:
        x_mix: mixed training data
        y_mix: mixed lables (soft)
        """
        distribution = self.find_distribution(y)
        orig_dis = distribution.copy()
        new_dis = [0] * len(distribution)
        x_mix = x
        y_mix = y
        n_batches = 0
        pb = tqdm(total=outside.steps)
        print("total steps: {}".format(outside.steps))
        for batch in outside.batch_loader():
            y_pred = model.predict(batch)
            indices, _, increase = self._find_good(y_pred, distribution)
            x_mix = np.concatenate([x_mix, batch[0][indices]], axis=0)
            y_mix = np.concatenate([y_mix, y_pred[indices]], axis=0)
            new_dis = [n + i for n, i in zip(new_dis, increase)]
            n_batches += 1
            pb.update(1)
            if n_batches == outside.steps:
                break
        if shuffle:
            randomed_idx = np.random.permutation(x_mix.shape[0])
            np.take(x_mix, randomed_idx, axis=0, out=x_mix)
            np.take(y_mix, randomed_idx, axis=0, out=y_mix)
        return x_mix, y_mix, orig_dis, new_dis

    def train_teacher(
        self,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        x_unlabeled,
        x_partial,
        y_partial,
        log_f,
        log_path,
        n_repeat,
    ):
        r""" Train linear model with Noisy Student and the inputs are balanced
        by the first teacher model
        model: the model to be trained
        x_train: training data
        y_train: labels of the training data
        x_test: testing data
        y_test: testing data labels
        x_unlabeled: unlabeled training data
        x_partial: partially labeled data
        y_partial: the partial labels
        log_f: logging file handler
        log_path: path to the logging directory
        n_repeat: times to re-train the model with balanced data
        =======================================================================
        return: the trained model, training histories
        """
        model = init_model(model)
        cb_list = callback_list(log_path, self.es_patience, model)
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
            y_pred_partial = model.predict(x_partial)
            y_pred_partial = fill_unlabeled(y_pred_partial, y_partial, normalize=True)
            # label outside data
            x_mix, y_mix, orig_dis, new_dis = self._predict_and_balance(
                model=model, outside=x_unlabeled, x=x_train, y=y_train, shuffle=True
            )
            # combine partially labeled and unlabeled
            x_mix = np.concatenate([x_partial, x_mix], axis=0)
            y_mix = np.concatenate([y_pred_partial, y_mix], axis=0)
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
            y_pred = model.predict(x_test)
            training_log(train_his, y_pred, y_test, log_f)
            self._plot_distribution_change(
                orig_dis,
                new_dis,
                name=str(model) + "_repeat" + str(i) + "_distribution.png",
            )

        return model, histories

    def train_student(
        self,
        student_model,
        teacher_model,
        x_train,
        y_train,
        x_test,
        y_test,
        x_unlabeled,
        x_partial,
        y_partial,
        log_f,
        log_path,
        n_repeat,
    ):
        r""" Train student linear model with Noisy Student
        student_model: the model to be trained
        teacher_model: trained model used to generate labels
        x_train: training data
        y_train: labels of the training data
        x_test: testing data
        y_test: testing data labels
        x_unlabeled: unlabeled training data
        x_partial: partially labeled data
        y_partial: the partial labels
        cb_list: callback list
        log_f: logging file handler
        log_path: path to the logging directory
        n_repeat: times to train the model
        =======================================================================
        return: the trained model, training histories
        """
        # label partially labeld data with the teacher model
        y_pred_partial = teacher_model.predict(x_partial)
        y_pred_partial = fill_unlabeled(y_pred_partial, y_partial, normalize=True)
        # balance training data with outside dataset
        x_mix, y_mix, _, _ = self._predict_and_balance(
            model=teacher_model, outside=x_unlabeled, x=x_train, y=y_train, shuffle=True
        )
        # combine partially labeled and unlabeled
        x_mix = np.concatenate([x_partial, x_mix], axis=0)
        y_mix = np.concatenate([y_pred_partial, y_mix], axis=0)
        # shuffle
        randomed_idx = np.random.permutation(x_mix.shape[0])
        np.take(x_mix, randomed_idx, axis=0, out=x_mix)
        np.take(y_mix, randomed_idx, axis=0, out=y_mix)
        # mixup
        if self.mixup is not None:
            x_mix, y_mix = self._mixup(x_mix, y_mix)
        # init model
        model = init_model(student_model)
        # callbacks
        cb_list = callback_list(log_path, self.es_patience, model)
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

        y_pred = model.predict(x_test)
        training_log(train_his, y_pred, y_test, log_f)

        # repeat training the model
        for i in range(n_repeat):
            log_f.write(
                "repeat training {}, {}/{}:\n".format(str(model), i + 1, n_repeat)
            )
            # label partially labled data
            y_pred_partial = model.predict(x_partial)
            y_pred_partial = fill_unlabeled(y_pred_partial, y_partial, normalize=True)
            # label unlabled
            x_mix, y_mix, orig_dis, new_dis = self._predict_and_balance(
                model=model, outside=x_unlabeled, x=x_train, y=y_train, shuffle=True
            )
            # combine partially labeled and unlabeled
            x_mix = np.concatenate([x_partial, x_mix], axis=0)
            y_mix = np.concatenate([y_pred_partial, y_mix], axis=0)
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
            y_pred = model.predict(x_test)
            training_log(train_his, y_pred, y_test, log_f)
            self._plot_distribution_change(
                orig_dis,
                new_dis,
                name=str(model) + "_repeat" + str(i) + "_distribution.png",
            )

        return model, histories

    def run_experiment(self):
        # load training and testing data
        (
            x_train,
            y_train,
            x_test,
            y_test,
            x_partial,
            y_partial,
            outside_data_loader,
        ) = self.load_data()
        # specific for ChEMBL24 dataset
        outside_data_loader.set_dataset("/ChEMBL/ECFP", shuffle=True, infinite=True)
        outside_data_loader.set_batch_size(self.batch_size)
        # open log
        log_f, log_path = self.open_log_(self.log_path)
        # train the teacher model
        trained_model, histories = self.train_teacher(
            model=Linear_S,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_unlabeled=outside_data_loader,
            x_partial=x_partial,
            y_partial=y_partial,
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
                x_unlabeled=outside_data_loader,
                x_partial=x_partial,
                y_partial=y_partial,
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
    parser = LMMixupOutsideDataArgs()
    args = parser.parse_args()
    experiment = ExpLinBalLargeOutsideExploitPartialNoMixup(
        data_path=args.data_path,
        outside_data_path=args.outside_path,
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
