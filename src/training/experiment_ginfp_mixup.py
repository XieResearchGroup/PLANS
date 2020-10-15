###############################################################
# Train HMLC, HMLC_M, HMLC_L models with Noisy Student method #
###############################################################

import tensorflow as tf

from ..data_loaders.json_loader import JsonLoader
from ..utils.label_convertors import convert2vec, partial2onehot
from .training_args import LMMixupArgs
from .experiment_linear_exploit_partial import ExperimentLinearExploitPartial


class ExperimentLinearGinFP(ExperimentLinearExploitPartial):
    def load_data(self):
        data_loader = JsonLoader(self.data_path)
        x_train, y_train, x_test, y_test = data_loader.load_data(ratio=0.7)
        y_train = convert2vec(y_train, dtype=float)
        y_test = convert2vec(y_test, dtype=float)  # for evaluation after training

        if self.mixup is not None:
            x_train, y_train = self._mixup(x_train, y_train)
        data_unlabeled = data_loader.load_unlabeled()
        x_unlabeled = data_unlabeled[:, 0]
        y_partial = data_unlabeled[:, 1]
        for i, label in enumerate(y_partial):
            y_partial[i] = partial2onehot(label)
        return x_train, y_train, x_test, y_test, x_unlabeled, y_partial


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
        rand_seed=args.rand_seed,
    )
    experiment.run_experiment()
