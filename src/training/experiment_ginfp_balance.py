#####################################################################################
# Experiment with GIN fingerprints. Noisy Student only. Exploit partial labels but  #
# no mixup.                                                                         #
#####################################################################################


import tensorflow as tf
import numpy as np

from ..data_loaders.json_loader import JsonLoader
from ..data_loaders.hdf5_loader import HDF5Loader
from ..utils.label_convertors import convert2vec, partial2onehot, multilabel2onehot
from ..models.linear import Linear_S, Linear_M, Linear_L
from .training_args import LMMixupOutsideDataArgs
from .experiment_linear_chembl_balance_partial_no_mixup import (
    ExpLinBalLargeOutsideExploitPartialNoMixup,
)


class ExperimentLinearGinFPBalance(ExpLinBalLargeOutsideExploitPartialNoMixup):
    def load_data(self):
        data_loader = JsonLoader(self.data_path)
        x_train, y_train, x_test, y_test = data_loader.load_data(
            ratio=0.7, shuffle=True
        )
        y_train = np.stack([multilabel2onehot(l, return_type="vec") for l in y_train]).astype(np.float)
        y_test = np.stack([multilabel2onehot(l, return_type="vec") for l in y_test]).astype(np.float)
        if self.mixup is not None:
            x_train, y_train = self._mixup(x_train, y_train)
        x_unlabeled, y_partial = data_loader.load_unlabeled()
        for i, label in enumerate(y_partial):
            y_partial[i] = partial2onehot(label)
            
        outside_data_loader = HDF5Loader(self.outside_data_path, "r")
        return (
            x_train,
            y_train,
            x_test,
            y_test,
            x_unlabeled,
            y_partial,
            outside_data_loader,
        )

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
        outside_data_loader.set_dataset("GINFP", shuffle=True, infinite=True)
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
    gpus = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    parser = LMMixupOutsideDataArgs()
    args = parser.parse_args()
    experiment = ExperimentLinearGinFPBalance(
        data_path=args.data_path,
        outside_data_path=args.outside_path,
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
