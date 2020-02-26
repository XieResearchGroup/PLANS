from functools import partial

import tensorflow as tf

from .experiment_base import ExperimentBase
from .training_args import LMMixupArgs
from .train_model import ns_linear_student_model
from .train_model import predict_with_multiteacher_and_mix
from ..utils.training_utils import init_model, callback_list
from ..models.linear import Linear_S, Linear_M, Linear_L
from ..utils.label_convertors import convert2vec, multivec2onehot
from ..data_loaders.cvs_loader import CVSLoader


class ExperimentSeparateModels(ExperimentBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_data(self, data_path):
        data_loader = CVSLoader(data_path)
        x_train, y_train, x_test, y_test = data_loader.load_data(
            ["ECFP", "Label"],
            ratio=0.7,
            shuffle=True)
        convert2vec_float = partial(convert2vec, dtype=float)
        x_train, y_train, x_test, y_test = list(
            map(convert2vec_float, [x_train, y_train, x_test, y_test]))
        if self.mixup is not None:
            x_train, y_train = self._mixup(x_train, y_train)
        x_unlabeled = data_loader.load_unlabeled(["ECFP", "Label"])
        x_unlabeled = convert2vec(x_unlabeled[:, 0], dtype=float)
        y_train_oh = multivec2onehot(y_train)
        y_test_oh = multivec2onehot(y_test)
        return x_train, y_train, x_test, y_test, x_unlabeled, \
            y_train_oh, y_test_oh

    def train_first_student(self,
                            student_model,
                            teacher_models,
                            x_train,
                            y_train,
                            x_test,
                            y_test,
                            x_pred,
                            batch_size,
                            epochs,
                            log_f,
                            log_path,
                            n_repeat):
        x_mix, y_mix = predict_with_multiteacher_and_mix(teacher_models,
                                                         x_pred,
                                                         x_train,
                                                         y_train,
                                                         shuffle=True)
        x_mix, y_mix = self._mixup(x_mix, y_mix)
        # init model
        model = init_model(student_model)
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
            n_repeat=n_repeat
        )
        return trained_model, histories

    def run_experiment(self):
        # load training and testing data
        x_train, y_train, x_test, y_test, x_unlabeled, y_train_oh, y_test_oh =\
            self.load_data(self.data_path)
        # open log
        log_f, log_path = self.open_log_(self.log_path)
        # train five separated teacher models
        trained_models = list()
        model_histories = list()
        teacher_model = partial(Linear_S, out_len=1)
        for i in range(y_train.shape[1]):
            trained_model, histories = self.train_teacher(
                model=teacher_model,
                x_train=x_train,
                y_train=y_train[:, i].reshape(-1, 1),
                x_test=x_test,
                y_test=y_test[:, i].reshape(-1, 1),
                x_pred=x_unlabeled,
                batch_size=self.batch_size,
                epochs=self.epochs,
                log_f=log_f,
                log_path=log_path,
                n_repeat=self.n_repeat
            )
            trained_models.append(trained_model)
            model_histories.append(histories)

        # train first student model
        trained_model, histories = self.train_first_student(
            student_model=Linear_S,
            teacher_models=trained_models,
            x_train=x_train,
            y_train=y_train_oh,
            x_test=x_test,
            y_test=y_test_oh,
            x_pred=x_unlabeled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            log_f=log_f,
            log_path=log_path,
            n_repeat=self.n_repeat
        )
        # log results
        self.log_training(trained_model, histories, log_path)
        # free memory
        tf.reset_default_graph()

        # train other students
        for student in [Linear_M, Linear_L]:
            trained_model, histories = self.train_student(
                student_model=student,
                teacher_model=trained_model,
                x_train=x_train,
                y_train=y_train_oh,
                x_test=x_test,
                y_test=y_test_oh,
                x_pred=x_unlabeled,
                batch_size=self.batch_size,
                epochs=self.epochs,
                log_f=log_f,
                log_path=log_path,
                n_repeat=self.n_repeat
            )
            # log results
            self.log_training(student, histories, log_path)

        log_f.close()


if __name__ == "__main__":
    parser = LMMixupArgs()
    args = parser.parse_args()
    experiment = ExperimentSeparateModels(
        data_path=args.data_path,
        log_path=args.log_path,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_repeat=args.repeat,
        mixup=args.mixup,
        mixup_repeat=args.mixup_repeat
    )
    experiment.run_experiment()
