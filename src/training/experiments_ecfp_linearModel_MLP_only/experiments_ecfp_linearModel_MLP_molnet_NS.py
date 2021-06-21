from ...models.linear import Linear_S, Linear_M, Linear_L
from ..experiment_base import ExperimentBase
from ..train_model import ns_linear_teacher_model, ns_linear_student_model
from ..train_model import predict_and_mix
from ..training_args import LMMixupArgs
from ...utils.training_utils import init_model, callback_list


class Experiment(ExperimentBase):
    def train_teacher(
        self,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        x_pred,
        batch_size,
        epochs,
        log_f,
        log_path,
        n_repeat,
        activation="sigmoid",
        loss="categorical_crossentropy",
        out_len=32,
    ):
        model = init_model(
            model,
            drop_rate=self.drop_rate,
            loss=loss,
            out_len=out_len,
            activation=activation,
        )
        cb_list = callback_list(log_path, self.es_patience, model)
        trained_model, histories = ns_linear_teacher_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_pred=x_pred,
            batch_size=batch_size,
            epochs=epochs,
            cb_list=cb_list,
            log_f=log_f,
            log_path=log_path,
            n_repeat=n_repeat,
        )
        return trained_model, histories

    def train_student(
        self,
        student_model,
        teacher_model,
        x_train,
        y_train,
        x_test,
        y_test,
        x_pred,
        batch_size,
        epochs,
        log_f,
        log_path,
        n_repeat,
        activation="sigmoid",
        loss="categorical_crossentropy",
        out_len=32,
    ):
        x_mix, y_mix = predict_and_mix(
            teacher_model, x_pred, x_train, y_train, shuffle=True
        )
        x_mix, y_mix = self._mixup(x_mix, y_mix)
        # init model
        model = init_model(
            student_model,
            drop_rate=self.drop_rate,
            loss=loss,
            out_len=out_len,
            activation=activation,
        )
        # callbacks
        cb_list = callback_list(log_path, self.es_patience, model)
        # fit Linear_M model to mixed dataset
        trained_model, histories = ns_linear_student_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_mix=x_mix,
            y_mix=y_mix,
            x_test=x_test,
            y_test=y_test,
            x_pred=x_pred,
            batch_size=batch_size,
            epochs=epochs,
            cb_list=cb_list,
            log_f=log_f,
            log_path=log_path,
            n_repeat=n_repeat,
        )
        return trained_model, histories

    def run_experiment(self):
        # load training and testing data
        x_train, y_train, x_test, y_test, x_unlabeled = self.load_data(
            ["ECFP", "Label"]
        )
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
    )
    experiment.run_experiment()
