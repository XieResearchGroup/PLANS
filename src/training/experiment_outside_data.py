###############################################################
# Train HMLC, HMLC_M, HMLC_L models with Noisy Student method #
# and the unlabeled data from sources other than CYP450       #
###############################################################

import os
from datetime import datetime

import numpy as np
import pandas as pd

from ..data_loaders.cvs_loader import CVSLoader
from ..utils.label_convertors import convert2vec, convert2hier
from .train_model import noisy_student_L
from .training_args import TrainingArgs


def main(data_path: str,
         outside_data_path: list,
         DataLoader=CVSLoader,
         columns=["ECFP", "Label"],
         learning_rate=0.001,
         drop_rate=0.3,
         batch_size=128,
         epochs=100,
         es_patience=5,
         log_path="../logs",
         if_hard=False,
         comment=None,
         unlabeled_weight=1.0):
    # Data
    cyp_data_loader = DataLoader(data_path)
    x_train, y_train, x_test, y_test = cyp_data_loader.load_data(columns)

    x_train = convert2vec(x_train, dtype=float)
    y_train = convert2hier(y_train, dtype=float)

    x_test = convert2vec(x_test, dtype=float)
    y_val = convert2hier(y_test, dtype=float)  # for validation during training
    y_eval = convert2vec(y_test, dtype=int)    # for evaluation after training

    data_pred = cyp_data_loader.load_unlabeled(["ECFP", "Label"])

    # Open log
    now = datetime.now()
    timestamp = now.strftime(r"%Y%m%d_%H%M%S")
    log_path = os.path.sep.join(log_path.split("/"))
    log_path = os.path.join(log_path, timestamp)
    os.makedirs(log_path, exist_ok=True)
    log_f_path = os.path.join(log_path, "logs.txt")
    log_f = open(log_f_path, "w")

    # train control model
    log_f.write("#" * 40 + "\n")
    log_f.write("Noisy Student L w/o extra data" + "\n")
    noisy_student_L(
        x_train=x_train,
        y_train=y_train,
        unlabeled_weight=1,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval,
        data_pred=data_pred,
        learning_rate=learning_rate,
        drop_rate=drop_rate,
        batch_size=batch_size,
        epochs=epochs,
        es_patience=es_patience,
        log_path=log_path,
        log_fh=log_f,
        comment=comment)

    # add extra dataset into training set
    for path in outside_data_path:
        outside_df = pd.read_csv(path, nrows=100000)
        outside_df["Label"] = np.array(["_"*5]*outside_df.shape[0])
        data_pred = np.concatenate(
            [data_pred, outside_df[["ECFP", "Label"]].to_numpy()],
            axis=0)

    # train Noisy Student L with extra data
    log_f.write("#" * 40 + "\n")
    log_f.write("Noisy Student L w/ extra data" + "\n")
    noisy_student_L(
        x_train=x_train,
        y_train=y_train,
        unlabeled_weight=1,
        x_test=x_test,
        y_val=y_val,
        y_eval=y_eval,
        data_pred=data_pred,
        learning_rate=learning_rate,
        drop_rate=drop_rate,
        batch_size=batch_size,
        epochs=epochs,
        es_patience=es_patience,
        log_path=log_path,
        log_fh=log_f,
        comment=comment)

    log_f.write("#" * 40 + "\n")
    log_f.close()


if __name__ == "__main__":
    parser = TrainingArgs()
    args = parser.parse_args()
    main(**vars(args),
         outside_data_path=[
             "./data/DrugBank_smiles_fp.csv",
             "./data/ChEMBL24_smiles_fp.csv"
         ])
