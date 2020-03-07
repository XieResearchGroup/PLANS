from .experiment_base import ExperimentBase
from .training_args import LMMixupArgs
from ..models.linear import Linear_M


class ExperimentLinear(ExperimentBase):
    r""" Experiment that does not use Noisy Student method
    """

    def run_experiment(self):
        # load training and testing data
        x_train, y_train, x_test, y_test, _ = self.load_data()
        # open log
        log_f, log_path = self.open_log_(self.log_path)
        # train the teacher model without repeat
        trained_model, histories = self.train_teacher(
            model=Linear_M,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
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
    parser = LMMixupArgs()
    args = parser.parse_args()
    experiment = ExperimentLinear(
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
