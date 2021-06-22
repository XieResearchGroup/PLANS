from functools import partial

from ..experiment_base import ExperimentBase
from ...models.linear import Linear_S, Linear_M, Linear_L
from ..training_args import LMMixupArgs
from ...data_loaders.json_loader import JsonLoader
from ...utils.label_convertors import convert2vec


class ExperimentLinearGinFPNSNoPartial(ExperimentBase):
    def load_data(self):
        data_loader = JsonLoader(self.data_path, rand_seed=self.rand_seed)
        x_train, y_train, x_test, y_test = data_loader.load_data(
            ratio=0.7, shuffle=True
        )
        convert2vec_float = partial(convert2vec, dtype=float)
        x_train, y_train, x_test, y_test = list(
            map(convert2vec_float, [x_train, y_train, x_test, y_test])
        )
        if self.mixup is not None:
            x_train, y_train = self._mixup(x_train, y_train)
        x_unlabeled, _ = data_loader.load_unlabeled()
        return x_train, y_train, x_test, y_test, x_unlabeled

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
            batch_size=self.batch_size,
            epochs=self.epochs,
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
                batch_size=self.batch_size,
                epochs=self.epochs,
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
        drop_rate=args.drop_rate,
    )
    experiment.run_experiment()
