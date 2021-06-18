import numpy as np

from ..experiment_base import ExperimentBase
from ..training_args import LMMixupArgs
from ...models.linear import Linear_M
from ...data_loaders.json_loader import JsonLoader
from ...utils.label_convertors import multilabel2onehot


class Experiment(ExperimentBase):
    r""" Experiment that does not use Noisy Student method. Use GINFP as inputs. Use
    scaffold splitting to split training and testing data.
    """

    def load_data(self):
        data_loader = JsonLoader(self.data_path)
        x_train, y_train, x_test, y_test = data_loader.load_data(
            ratio=0.7, shuffle=True, scaffold_splitting=True
        )
        y_train = np.stack(
            [multilabel2onehot(l, return_type="vec") for l in y_train]
        ).astype(np.float)
        y_test = np.stack(
            [multilabel2onehot(l, return_type="vec") for l in y_test]
        ).astype(np.float)
        if self.mixup is not None:
            x_train, y_train = self._mixup(x_train, y_train)
        x_unlabeled, _ = data_loader.load_unlabeled()
        return x_train, y_train, x_test, y_test, x_unlabeled

    def run_experiment(self):
        # load training and testing data
        x_train, y_train, x_test, y_test, x_unlabeled = self.load_data()
        # open log
        log_f, log_path = self.open_log_(self.log_path)
        # train the teacher model without repeat
        trained_model, histories = self.train_teacher(
            model=Linear_M,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_pred=x_unlabeled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            log_f=log_f,
            log_path=log_path,
            n_repeat=0,
        )
        # log results
        self.log_training(trained_model, histories, log_path)

        log_f.write("best losses:\n {}\n".format(str(self.best_loss)))
        log_f.write("best accuracies:\n {}\n".format(str(self.best_acc)))
        log_f.close()


if __name__ == "__main__":
    parser = LMMixupArgs()
    args = parser.parse_args()
    experiment = Experiment(
        data_path=args.data_path,
        log_path=args.log_path,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_repeat=args.repeat,
        mixup=args.mixup,
        mixup_repeat=args.mixup_repeat,
        rand_seed=args.rand_seed,
        learning_rate=args.learning_rate,
    )
    experiment.run_experiment()
