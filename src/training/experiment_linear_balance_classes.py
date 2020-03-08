import os
from functools import partial

import numpy as np
import matplotlib.pyplot as plt

from .experiment_base import ExperimentBase
from ..models.linear import Linear_S, Linear_M, Linear_L
from .training_args import LMMixupOutsideDataArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec
from ..utils.training_utils import init_model, callback_list, training_log


class ExperimentLinearBalanced(ExperimentBase):

    def __init__(self,
                 data_path,
                 outside_data_path,
                 log_path,
                 es_patience,
                 batch_size,
                 epochs,
                 n_repeat,
                 rand_seed,
                 mixup=None,
                 mixup_repeat=None):
        super().__init__(data_path=data_path,
                         log_path=log_path,
                         es_patience=es_patience,
                         batch_size=batch_size,
                         epochs=epochs,
                         n_repeat=n_repeat,
                         rand_seed=rand_seed,
                         mixup=mixup,
                         mixup_repeat=mixup_repeat)
        self.outside_data_path = outside_data_path

    def load_data(self):
        data_loader = CVSLoader(self.data_path, rand_seed=self.rand_seed)
        outside_data_loader = CVSLoader(self.outside_data_path)
        x_train, y_train, x_test, y_test = data_loader.load_data(
            ["ECFP", "onehot_label"],
            ratio=0.7,
            shuffle=True)
        convert2vec_float = partial(convert2vec, dtype=float)
        x_train, y_train, x_test, y_test = list(
            map(convert2vec_float, [x_train, y_train, x_test, y_test]))
        if self.mixup is not None:
            x_train, y_train = self._mixup(x_train, y_train)
        x_cyp_unlabeled = data_loader.load_unlabeled(["ECFP", "onehot_label"])
        x_cyp_unlabeled = convert2vec(x_cyp_unlabeled[:, 0], dtype=float)
        x_outside = outside_data_loader.load_col("ECFP")
        x_outside = convert2vec(x_outside, dtype=float)
        x_unlabeled = np.concatenate([x_cyp_unlabeled, x_outside], axis=0)
        return x_train, y_train, x_test, y_test, x_unlabeled

    def find_distribution(self, labels):
        r""" Find the distribution of training data
        labels (numpy.array): training set labels.
        =======================================================================
        return (list): a classwise counts
        """
        distribution = [0] * labels.shape[1]
        for label in labels:
            distribution[np.argmax(label)] += 1
        return distribution

    def _find_good(self, predictions, distribution):
        orig = distribution.copy()
        increase = [0] * len(distribution)
        good_indices = list()
        max_count = max(distribution)
        max_index = np.argmax(distribution)
        for i, pred in enumerate(predictions):
            m_ind = np.argmax(pred)
            if m_ind != max_index and distribution[m_ind] < max_count:
                distribution[m_ind] += 1
                increase[m_ind] += 1
                good_indices.append(i)
        return good_indices, orig, increase

    def _predict_and_balance(self, model, x_pred, x, y, shuffle=True):
        r""" Make predictions with the given model and mix it with the existing
        training set.
        model (tf.keras.Model): the pretrained model to make predictions
        x_pred (np.array): inputs for the model
        x (np.arrapy): training set data
        y (np.array): training set label
        shuffle (bool): if shuffle after mixing the training and predicted data
        =======================================================================
        return:
        x_mix: mixed training data
        y_mix: mixed lables (soft)
        """
        y_pred = model.predict(x_pred)
        distribution = self.find_distribution(y)
        indices, orig_dis, new_dis = self._find_good(y_pred, distribution)
        x_mix = np.concatenate([x, x_pred[indices]], axis=0)
        y_mix = np.concatenate([y, y_pred[indices]], axis=0)
        if shuffle:
            randomed_idx = np.random.permutation(x_mix.shape[0])
            np.take(x_mix, randomed_idx, axis=0, out=x_mix)
            np.take(y_mix, randomed_idx, axis=0, out=y_mix)
        return x_mix, y_mix, orig_dis, new_dis

    def _plot_distribution_change(self, orig, new, name):
        fig, axes = plt.subplots(ncols=2, figsize=[12.8, 4.8])
        inds = list(range(len(orig)))
        y_max = max(orig) + 20
        axes[0].bar(inds, orig)
        axes[1].bar(inds, orig)
        axes[1].bar(inds, new, bottom=orig)
        for ax, title in zip(axes, ["original", "new"]):
            ax.set(ylim=[0, y_max],
                   xlabel="classes",
                   ylabel="Counts",
                   title=title)
        fig.savefig(os.path.join(self.log_path, name), dpi=300)

    def train_teacher(self,
                      model,
                      x_train,
                      y_train,
                      x_test,
                      y_test,
                      x_pred,
                      log_f,
                      log_path,
                      n_repeat):
        r""" Train linear model with Noisy Student and the inputs are balanced
        by the first teacher model
        model: the model to be trained
        x_train: training data
        y_train: labels of the training data
        x_test: testing data
        y_test: testing data labels
        x_pred: unlabeled training data
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
            validation_data=[x_test, y_test]
        )
        histories.append(train_his)

        y_pred = model.predict(x_test)
        training_log(train_his, y_pred, y_test, log_f)

        # repeat training the model
        for i in range(n_repeat):
            log_f.write(
                "repeat training {}, {}/{}:\n".format(
                    str(model), i+1, n_repeat))
            # label unlabled
            x_mix, y_mix, orig_dis, new_dis = self._predict_and_balance(
                model=model,
                x_pred=x_pred,
                x=x_train,
                y=y_train,
                shuffle=True
            )
            if self.mixup is not None:
                x_mix, y_mix = self._mixup(x_mix, y_mix,)
            # train model with the mixed data
            train_his = model.fit(
                x=x_mix,
                y=y_mix,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=cb_list,
                validation_data=[x_test, y_test]
            )
            histories.append(train_his)
            # log training history
            y_pred = model.predict(x_test)
            training_log(train_his, y_pred, y_test, log_f)
            self._plot_distribution_change(
                orig_dis,
                new_dis,
                name=str(model)+"_repeat"+str(i)+"_distribution.png")

        return model, histories

    def train_student(self,
                      student_model,
                      teacher_model,
                      x_train,
                      y_train,
                      x_test,
                      y_test,
                      x_pred,
                      log_f,
                      log_path,
                      n_repeat):
        r""" Train student linear model with Noisy Student
        student_model: the model to be trained
        teacher_model: trained model used to generate labels
        x_train: training data
        y_train: labels of the training data
        x_test: testing data
        y_test: testing data labels
        x_pred: unlabeled training data
        cb_list: callback list
        log_f: logging file handler
        log_path: path to the logging directory
        n_repeat: times to train the model
        =======================================================================
        return: the trained model, training histories
        """
        x_mix, y_mix, _, _ = self._predict_and_balance(teacher_model,
                                                       x_pred,
                                                       x_train,
                                                       y_train,
                                                       shuffle=True)
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
            validation_data=[x_test, y_test]
        )
        histories.append(train_his)

        y_pred = model.predict(x_test)
        training_log(train_his, y_pred, y_test, log_f)

        # repeat training the model
        for i in range(n_repeat):
            log_f.write(
                "repeat training {}, {}/{}:\n".format(
                    str(model), i+1, n_repeat))
            # label unlabled
            x_mix, y_mix, orig_dis, new_dis = self._predict_and_balance(
                model=model,
                x_pred=x_pred,
                x=x_train,
                y=y_train,
                shuffle=True
            )
            if self.mixup is not None:
                x_mix, y_mix = self._mixup(x_mix, y_mix)
            # train model with the mixed data
            train_his = model.fit(
                x=x_mix,
                y=y_mix,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=cb_list,
                validation_data=[x_test, y_test]
            )
            histories.append(train_his)
            # log training history
            y_pred = model.predict(x_test)
            training_log(train_his, y_pred, y_test, log_f)
            self._plot_distribution_change(
                orig_dis,
                new_dis,
                name=str(model)+"_repeat"+str(i)+"_distribution.png")

        return model, histories

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
                log_f=log_f,
                log_path=log_path,
                n_repeat=self.n_repeat
            )
            # log results
            self.log_training(trained_model, histories, log_path)

        log_f.write("best losses:\n {}\n".format(str(self.best_loss)))
        log_f.write("best accuracies:\n {}\n".format(str(self.best_acc)))
        log_f.close()


if __name__ == "__main__":
    parser = LMMixupOutsideDataArgs()
    args = parser.parse_args()
    experiment = ExperimentLinearBalanced(
        data_path=args.data_path,
        outside_data_path=args.outside_path,
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
