from functools import partial

import numpy as np

from .experiment_linear_model_mixup import ExperimentLinearMixup
from ..models.linear import Linear_S, Linear_M, Linear_L
from .training_args import LMMixupOutsideDataArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec


class ExperimentLinearOutsideMixup(ExperimentLinearMixup):

    def __init__(self,
                 data_path,
                 outside_data_path,
                 log_path,
                 es_patience,
                 batch_size,
                 epochs,
                 n_repeat,
                 mixup,
                 mixup_repeat,
                 rand_seed=None):
        super().__init__(data_path,
                         log_path,
                         es_patience,
                         batch_size,
                         epochs,
                         n_repeat,
                         mixup,
                         mixup_repeat,
                         rand_seed)
        self.outside_data_path = outside_data_path

    def load_data(self, data_path, outside_data_path):
        data_loader = CVSLoader(data_path, rand_seed=self.rand_seed)
        outside_data_loader = CVSLoader(outside_data_path)
        x_train, y_train, x_test, y_test = data_loader.load_data(
            ["ECFP", "onehot_label"],
            ratio=0.7,
            shuffle=True)
        convert2vec_float = partial(convert2vec, dtype=float)
        x_train, y_train, x_test, y_test = list(
            map(convert2vec_float, [x_train, y_train, x_test, y_test]))
        x_train, y_train = self._mixup(x_train, y_train)
        x_cyp_unlabeled = data_loader.load_unlabeled(["ECFP", "onehot_label"])
        x_cyp_unlabeled = convert2vec(x_cyp_unlabeled[:, 0], dtype=float)
        x_outside = outside_data_loader.load_col("ECFP")
        x_outside = convert2vec(x_outside, dtype=float)
        x_unlabeled = np.concatenate([x_cyp_unlabeled, x_outside], axis=0)
        return x_train, y_train, x_test, y_test, x_unlabeled

    def run_experiment(self):
        # load training and testing data
        x_train, y_train, x_test, y_test, x_unlabeled = self.load_data(
            self.data_path, self.outside_data_path)
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
    parser = LMMixupOutsideDataArgs()
    args = parser.parse_args()
    experiment = ExperimentLinearOutsideMixup(
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
