"""
Analyze the output file in .pk format. The y_preds are continuous float numbers. The
rests are integers.

The results are print out and written to a json file when the write() method is
executed.
"""

import os
import pickle as pk
from collections import defaultdict
from statistics import mean, stdev
import json

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score


def get_auc(predictions):
    return roc_auc_score(predictions["y"], predictions["y_pred"], average="micro")


def get_f1(predictions):
    return f1_score(predictions["y"], np.round(predictions["y_pred"]), average="micro",)


def get_precision(predictions):
    return precision_score(
        predictions["y"], np.round(predictions["y_pred"]), average="micro",
    )


def get_recall(predictions):
    return recall_score(
        predictions["y"], np.round(predictions["y_pred"]), average="micro",
    )


def get_scores(path, scores={}):
    with open(path, "rb") as f:
        predictions = pk.load(f)
    scores["auc"].append(get_auc(predictions))
    scores["F1"].append(get_f1(predictions))
    scores["recall"].append(get_recall(predictions))
    scores["precision"].append(get_precision(predictions))


class Statistics:
    def __init__(self, scores):
        self.scores = scores

    def parse_scores(self):
        statistics = {}
        for k, v in self.scores.items():
            statistics[k] = (mean(v), stdev(v))
        return statistics

    def write(self, path):
        outf = os.path.join(path, "statistics.json")
        with open(outf, "w") as f:
            json.dump(self.parse_scores(), f)


def analyze(path):
    scores = defaultdict(list)
    for experiment in os.scandir(path):
        if experiment.name == ".ipynb_checkpoints":
            continue
        if experiment.is_dir():
            get_scores(os.path.join(experiment.path, "predictions.pk"), scores)
    logger = Statistics(scores)
    print(logger.parse_scores())
    logger.write(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path to the log directory.")
    args = parser.parse_args()
    analyze(args.path)
