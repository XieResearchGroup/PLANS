from functools import partial

import numpy as np

from .experiment_base import ExperimentBase
from ..models.linear import Linear_S, Linear_M, Linear_L
from .train_model import ns_linear_teacher_model, ns_linear_student_model
from .train_model import predict_and_mix, plot_history
from .training_args import LMMixupArgs
from ..data_loaders.json_loader import JsonLoader
from ..utils.label_convertors import multilabel2onehot
from ..utils.mixup import mixup
from ..utils.training_utils import init_model, callback_list, open_log
from ..utils.training_utils import find_best


class ExperimentLinearGinFPNSNoPartial(ExperimentBase):

    def load_data(self):
        data_loader = JsonLoader(self.data_path, rand_seed=self.rand_seed)
        x_train, y_train, x_test, y_test = data_loader.load_data(
            ratio=0.7, shuffle=True
        )
        y_train = np.stack([multilabel2onehot(l, return_type="vec") for l in y_train]).astype(np.float)
        y_test = np.stack([multilabel2onehot(l, return_type="vec") for l in y_test]).astype(np.float)
        if self.mixup is not None:
            x_train, y_train = self._mixup(x_train, y_train)
        x_unlabeled, _ = data_loader.load_unlabeled()
        return x_train, y_train, x_test, y_test, x_unlabeled


if __name__ == "__main__":
    parser = LMMixupArgs()
    args = parser.parse_args()
    experiment = ExperimentLinearGinFPNSNoPartial(
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
        drop_rate=args.drop_rate
    )
    experiment.run_experiment()
