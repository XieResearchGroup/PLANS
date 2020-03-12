from functools import partial

import numpy as np

from .experiment_base import ExperimentBase
from ..models.linear import Linear_S, Linear_M, Linear_L
from .training_args import LMMixupArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec


class ExperimentLinearOptMixup(ExperimentBase):

    def _opt_mixup(self,
                   coef,
                   samp,
                   label,
                   repeat):
        """ mixup two samples
        coef: Beta distribution coefficient, alpha=beta=coef
        samp: samples to be mixed
        label: label of the samples to be mixed
        repeat (int): times to repeat the inputs
        """
        # reshape labels if neccesary
        if len(label.shape) == 1:
            label = label.reshape(-1, 1)

        # remove negatives
        indices = np.argmax(samp, axis=1) != 0
        samp = samp[indices]
        label = label[indices]

        # repeat
        samp = np.repeat(samp, repeat, axis=0)
        label = np.repeat(label, repeat, axis=0)

        # shuffle inputs
        np.random.seed(1729)
        indices_1 = np.random.permutation(samp.shape[0])
        np.random.seed(8901)
        indices_2 = np.random.permutation(samp.shape[0])
        samp1 = np.take(samp, indices_1, axis=0)
        label1 = np.take(label, indices_1, axis=0)
        samp = np.take(samp, indices_2, axis=0)
        label = np.take(label, indices_2, axis=0)

        # sample lambda from beta distribution
        lamb = np.random.beta(coef, coef, (samp1.shape[0], 1))
        print("lamb shape: {}".format(lamb.shape))

        # mixup samples
        sample = lamb * samp1 + (1-lamb) * samp
        label = lamb * label1 + (1-lamb) * label

        return sample, label

    def _mixup(self, x, y):
        x_mix, y_mix = self._opt_mixup(
            self.mixup, x, y, repeat=self.mixup_repeat)
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

    def run_experiment(self):
        # load training and testing data
        x_train, y_train, x_test, y_test, x_unlabeled = self.load_data(
            self.data_path)
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
    experiment = ExperimentLinearOptMixup(
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
