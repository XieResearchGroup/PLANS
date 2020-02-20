import os

from functools import partial
from datetime import datetime

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

from ..models.linear import Linear_S
from .train_model import predict_and_mix
from .training_args import TrainingArgs
from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec


def main(data_path, log_path, es_patience, batch_size, epochs):

    convert2vec_float = partial(convert2vec, dtype=float)
    # data
    data_loader = CVSLoader(data_path)
    x_train, y_train, x_test, y_test = data_loader.load_data(
        ["ECFP", "onehot_label"],
        ratio=0.7,
        shuffle=True)
    x_train, y_train, x_test, y_test = list(
        map(convert2vec_float, [x_train, y_train, x_test, y_test]))

    # Open log
    now = datetime.now()
    timestamp = now.strftime(r"%Y%m%d_%H%M%S")
    log_path = os.path.sep.join(log_path.split("/"))
    log_path = os.path.join(log_path, timestamp)
    os.makedirs(log_path, exist_ok=True)
    log_f_path = os.path.join(log_path, "logs.txt")
    log_f = open(log_f_path, "w")

    # init model
    model = Linear_S()
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["acc"])
    tbcb = TensorBoard(log_path)
    escb = EarlyStopping(
        "val_acc",
        patience=es_patience,
        restore_best_weights=True,
        mode="max"
    )
    lrcb = LearningRateScheduler(model.scheduler)

    # fit
    train_his = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tbcb, escb, lrcb],
        validation_data=[x_test, y_test]
    )

    predictions = np.argmax(model.predict(x_test), axis=1)
    truth = np.argmax(y_test, axis=1)
    # log training history
    for k, v in train_his.history.items():
        log_f.write(k+"\n")
        log_f.write(str(v)[1:-1]+"\n\n")
    log_f.write("="*80+"\n")
    for pred, tr in zip(predictions, truth):
        log_f.write(str(pred)+" "+str(tr)+"\n")
    log_f.write("="*80+"\n")

    # repeat training the first model
    x_unlabeled = data_loader.load_unlabeled(["ECFP", "onehot_label"])
    x_unlabeled = convert2vec(x_unlabeled[:, 0], dtype=float)
    n_repeats = 4
    for i in range(n_repeats):
        log_f.write("repeat training {}:\n".format(i))
        # label unlabled
        x_mix, y_mix = predict_and_mix(
            model=model,
            x_pred=x_unlabeled,
            x=x_train,
            y=y_train,
            shuffle=True
        )
        # train model with the mixed data
        train_his = model.fit(
            x=x_mix,
            y=y_mix,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[tbcb, escb, lrcb],
            validation_data=[x_test, y_test]
        )
        # log training history
        predictions = np.argmax(model.predict(x_test), axis=1)
        truth = np.argmax(y_test, axis=1)
        for k, v in train_his.history.items():
            log_f.write(k+"\n")
            log_f.write(str(v)[1:-2]+"\n\n")
        log_f.write("="*80+"\n")
        for pred, tr in zip(predictions, truth):
            log_f.write(str(pred)+" "+str(tr)+"\n")
        log_f.write("="*80+"\n")

    log_f.close()

    return model


if __name__ == "__main__":
    parser = TrainingArgs()
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        log_path=args.log_path,
        es_patience=args.es_patience,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
