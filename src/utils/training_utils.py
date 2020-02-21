from datetime import datetime
import os

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np


def init_model(Model):
    model = Model()
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["acc"]
    )
    return model


def callback_list(log_path, es_patience, model):
    tbcb = TensorBoard(log_path)
    escb = EarlyStopping(
        "val_acc",
        patience=es_patience,
        restore_best_weights=True,
        mode="max"
    )
    lrcb = LearningRateScheduler(model.scheduler)
    cb_list = [tbcb, escb, lrcb]
    return cb_list


def training_log(train_his, y_pred, y_truth, log_f):
    predictions = np.argmax(y_pred, axis=1)
    truth = np.argmax(y_truth, axis=1)
    # log training history
    for k, v in train_his.history.items():
        log_f.write(k+"\n")
        log_f.write(str(v)[1:-1]+"\n\n")
    log_f.write("="*80+"\n")
    for pred, tr in zip(predictions, truth):
        log_f.write(str(pred)+" "+str(tr)+"\n")
    log_f.write("="*80+"\n")


def open_log(log_path):
    now = datetime.now()
    timestamp = now.strftime(r"%Y%m%d_%H%M%S")
    log_path = os.path.sep.join(log_path.split("/"))
    log_path = os.path.join(log_path, timestamp)
    os.makedirs(log_path, exist_ok=True)
    log_f_path = os.path.join(log_path, "logs.txt")
    log_f = open(log_f_path, "w")
    return log_f, log_path


def find_best(histories, field, mode):
    r""" Find the best value in the histories
    histories (list): list of keras History objects
    field (str): the field to find in the history
    mode (str): "min" or "max". The method to find the best value in the field
    ===========================================================================
    return (list): list of best values in the History objects
    """
    method = {"min": min, "max": max}
    find = method[mode]
    bests = list()
    for his in histories:
        bests.append(find(his.history[field]))
    return bests
