from functools import partial

import numpy as np
from sklearn.metrics import accuracy_score

from .experiment_base import ExperimentBase
from .training_args import LMMixupArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..models.linear import Linear_S, Linear_M
from ..utils.label_convertors import convert2vec


class ExperimentTwoModels(ExperimentBase):

    def load_data(self):
        data_loader = CVSLoader(self.data_path)
        x_train, y_onehot_train, y_2class_train, \
            x_test, y_onehot_test, y_2class_test = data_loader.load_data(
                ["ECFP", "onehot_label", "two_class_label"],
                ratio=0.7,
                shuffle=True)
        convert2vec_float = partial(convert2vec, dtype=float)
        x_train, y_onehot_train, x_test, y_onehot_test = list(
            map(
                convert2vec_float,
                [x_train, y_onehot_train, x_test, y_onehot_test]
            )
        )
        y_2class_train = np.expand_dims(y_2class_train, 1)
        y_2class_test = np.expand_dims(y_2class_test, 1)
        if self.mixup is not None:
            x_onehot_train, y_onehot_train = self._mixup(
                x_train, y_onehot_train)
        else:
            x_onehot_train = x_train
        x_unlabeled = data_loader.load_unlabeled(["ECFP", "onehot_label"])
        x_unlabeled = convert2vec(x_unlabeled[:, 0], dtype=float)
        return x_train, x_onehot_train, y_onehot_train, y_2class_train, \
            x_test, y_onehot_test, y_2class_test, x_unlabeled

    def _combine_predictions(self, binary, multi):
        results = np.zeros((binary.shape[0],), dtype=np.int)
        for i, (b, m) in enumerate(zip(binary, multi)):
            if b < 0.5:
                results[i] = 0
            else:
                results[i] = np.argmax(m) + 1
        return results

    def run_experiment(self):
        # load training and testing data
        x_train, x_onehot_train, y_onehot_train, y_2class_train, x_test, \
            y_onehot_test, y_2class_test, x_unlabeled = self.load_data()
        # open log
        log_f, log_path = self.open_log_(self.log_path)
        # train the 2-class model
        print("Training 2-class model...")
        two_class_model = partial(Linear_S, out_len=1)
        trained_2class_model, histories = self.train_teacher(
            model=two_class_model,
            x_train=x_train,
            y_train=y_2class_train,
            x_test=x_test,
            y_test=y_2class_test,
            x_pred=x_unlabeled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            log_f=log_f,
            log_path=log_path,
            n_repeat=self.n_repeat
        )
        # log results
        self.log_training(trained_2class_model, histories, log_path)

        # train the other classes model
        print("Training multi-class model...")
        multi_class_model = partial(Linear_M, out_len=31)
        trained_onehot_model, histories = self.train_teacher(
            model=multi_class_model,
            x_train=x_onehot_train,
            y_train=y_onehot_train[:, 1:],
            x_test=x_test,
            y_test=y_onehot_test[:, 1:],
            x_pred=x_unlabeled,
            batch_size=self.batch_size,
            epochs=self.epochs,
            log_f=log_f,
            log_path=log_path,
            n_repeat=self.n_repeat
        )
        # log results
        self.log_training(trained_onehot_model, histories, log_path)

        # evaluation
        binary_pred = trained_2class_model.predict(x_test)
        multi_pred = trained_onehot_model.predict(x_test)
        combined_pred = self._combine_predictions(binary_pred, multi_pred)
        y_true = np.argmax(y_onehot_test, axis=1)
        log_f.write("@final_evaluation\n")
        for pred, true in zip(combined_pred, y_true):
            log_f.write(str(pred)+" "+str(true)+"\n")
        log_f.write("="*80+"\n")

        acc_score = accuracy_score(y_true, combined_pred)
        log_f.write("accuracy score: \n{}\n".format(acc_score))

        log_f.write("best losses:\n{}\n".format(str(self.best_loss)))
        log_f.write("best accuracies:\n{}\n".format(str(self.best_acc)))
        log_f.close()


if __name__ == "__main__":
    parser = LMMixupArgs()
    args = parser.parse_args()
    experiment = ExperimentTwoModels(
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
