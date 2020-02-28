import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class BaseEvaluator:

    def __init__(self, log_file_path, mode="r"):
        self.log_f = open(log_file_path, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log_f.close()

    def __del__(self):
        if not self.log_f.closed:
            self.log_f.close()


class TrainingLogEvaluator(BaseEvaluator):

    def get_predictions(self):
        r""" Readout the predictions from the log file.
        =======================================================================
        return (list): list of the prediction results in the log file.
        """
        results = list()
        line = self.log_f.readline()
        while line:
            if line.startswith("@prediction-truth"):
                result = list()
                line = self.log_f.readline()
                while not line.startswith("="):
                    result.append(tuple(map(int, line.split())))
                    line = line = self.log_f.readline()
                results.append(result)
            line = self.log_f.readline()
        return results

    @property
    def _results(self):
        try:
            return self._results_
        except AttributeError:
            self._results_ = self.get_predictions()
            return self._results_

    def accuracy_scores(self):
        r""" Calculate accuracy scores from the results.
        =======================================================================
        return (list): list of accuracy scores calculated from the results.
        """
        accuracies = list()
        for rst in self._results:
            counter = 0
            for pred, truth in rst:
                if pred == truth:
                    counter += 1
            acc = float(counter) / len(rst)
            accuracies.append(acc)
        return accuracies

    def _int2multilabel(self, int_, label_len):
        # get the str representation of the binarized label
        bin_str = bin(int_)[2:]  # remove "0b"
        # construct the multilabel string
        padding = label_len - len(bin_str)
        if padding < 0:
            raise ValueError("label_len is too small for the results.")
        vec_str = padding * "0" + bin_str
        # convert string to np.array
        vec = np.array(list(map(int, list(vec_str))))
        return vec

    def _r2m(self, result, label_len=None):
        r""" int number result to multilabel
        inputs:
        result (list): list of (prediction, truth) tuples
        label_len (int): length of the returned labels, raise ValueError if
                         the length is too short for the result
        =======================================================================
        return (list): list of [predictions, truths]
        """
        predicts = np.zeros((len(result), label_len))
        truths = np.zeros((len(result), label_len))
        for i, (pred, truth) in enumerate(result):
            predicts[i, :] = self._int2multilabel(pred, label_len)
            truths[i, :] = self._int2multilabel(truth, label_len)
        return [predicts, truths]

    def _results2multilabel(self, label_len):
        multilabeled = list()
        for rst in self._results:
            ml = self._r2m(rst, label_len)
            multilabeled.append(ml)
        return multilabeled

    def precision_recall_fbeta_scores(self,
                                      classes=5,
                                      beta=1,
                                      average="micro"):
        r""" Calculate precision scores from the results.
        =======================================================================
        return (list): list of precision scores calculated from the results.
        """
        precisions = list()
        recalls = list()
        fbetas = list()
        for rst in self._results2multilabel(classes):
            pc, rc, fbeta, _ = precision_recall_fscore_support(
                rst[0], rst[1], average=average)
            precisions.append(pc)
            recalls.append(rc)
            fbetas.append(fbeta)
        return precisions, recalls, fbetas
