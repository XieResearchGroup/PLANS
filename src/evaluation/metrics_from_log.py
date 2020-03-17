import json
import statistics as stat

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class BaseEvaluator:

    def __init__(self, log_file_path, mode="r"):
        self.log_f = open(log_file_path, mode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.log_f.close()

    def __del__(self):
        if hasattr(self, "log_f") and not self.log_f.closed:
            self.log_f.close()


class TrainingLogEvaluator(BaseEvaluator):

    def get_predictions(self):
        r""" Readout the predictions from the log file.
        =======================================================================
        return (list): list of the prediction results in the log file.
        """
        # reset file pointer
        self.log_f.seek(0, 0)
        results = list()
        line = self.log_f.readline()
        while line:
            if line.startswith("@prediction-truth"):
                result = list()
                line = self.log_f.readline()
                while not line.startswith("="):
                    result.append(tuple(map(int, line.split())))
                    line = self.log_f.readline()
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

    def roc_auc(self, classes=5, average="micro"):
        r""" Calculate ROC-AUC from the results
        =======================================================================
        return (list)): ROC-AUC scores
        """
        scores = list()
        for rst in self._results2multilabel(classes):
            score = roc_auc_score(rst[0], rst[1], average=average)
            scores.append(score)
        return scores

    def classwise_hits_count_one_result(self, result, classes=32):
        r""" Count the correct, incorrect and missed predictions.
        result (list): list of (prediction, truth) pairs
        classes (int): number of classes
        =======================================================================
        return (list, list, list): number of correct, incorrect, and missed
            counts.
        """
        corrects = [0] * classes
        incorrects = [0] * classes
        missed = [0] * classes
        for pred, truth in result:
            if pred == truth:
                corrects[truth] += 1
            else:
                incorrects[pred] += 1
                missed[truth] += 1
        return corrects, incorrects, missed

    def classwise_hits_count(self, classes=32):
        counts = list()
        for rst in self._results:
            counts.append(self.classwise_hits_count_one_result(rst, classes))
        return counts

    def plot_classwise_prediction_bars(self,
                                       ylim=None,
                                       index=-1,
                                       classes=32,
                                       mask=None,
                                       save_name=None,
                                       show=True):
        r""" Plot bar diagram based on prediction values.
        index (int): the index of result in the results list. Default is -1.
        classes (int): classes in the plot. Default is 32.
        mask (list): list of the booleans to mask the bars. Default is None.
        """
        fig, axe = plt.subplots()
        ind = list(range(classes))
        counts = self.classwise_hits_count()
        corrects = counts[index][0]
        misses = counts[index][2]
        if mask is not None:
            for i, m in enumerate(mask):
                if not m:
                    corrects[i] = 0
                    misses[i] = 0
        axe.bar(ind, corrects, label="Correct")
        axe.bar(
            ind, misses, bottom=corrects, label="Missed")
        axe.legend()
        axe.set(xlabel="Classes", ylabel="Counts", ylim=ylim)
        if save_name is not None:
            fig.savefig(save_name, dpi=300)
        if show:
            fig.show()

    def plot_classwise_correct_and_incorrect_bars(self,
                                                  ylim=None,
                                                  index=-1,
                                                  classes=32,
                                                  mask=None,
                                                  save_name=None,
                                                  show=True):
        r""" Plot bar diagram based on correct and incorrect predicted values.
        index (int): the index of result in the results list. Default is -1.
        classes (int): classes in the plot. Default is 32.
        mask (list): list of the booleans to mask the bars. Default is None.
        """
        fig, axe = plt.subplots()
        ind = list(range(classes))
        counts = self.classwise_hits_count()
        corrects = counts[index][0]
        incorrects = counts[index][1]
        if mask is not None:
            for i, m in enumerate(mask):
                if not m:
                    corrects[i] = 0
                    incorrects[i] = 0
        axe.bar(ind, corrects, label="Correct")
        axe.bar(
            ind, incorrects, bottom=corrects, label="Incorrect", color="red")
        axe.legend()
        axe.set(xlabel="Classes", ylabel="Counts", ylim=ylim)
        if save_name is not None:
            fig.savefig(save_name, dpi=300)
        if show:
            fig.show()

    def get_best_acc(self, method=max):
        r""" Get the best validate accuracies from the log file
        method: Callable used to find the best value in the result list.
            Default is max.
        """
        self.log_f.seek(0, 0)
        line = self.log_f.readline()
        while not line.startswith("best accuracies"):  # Find the results
            line = self.log_f.readline()
        line = self.log_f.readline()  # Read the results
        accuracies = json.loads(line.replace("'", '"'))
        bests = dict()
        for k, v in accuracies.items():
            bests[k] = method(v)
        return bests


class BaseCalculator:

    def __init__(self, evaluators, output_path, mode="w"):
        self.evaluators = evaluators
        self.output_f = open(output_path, mode, encoding="utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.output_f.close()

    def __del__(self):
        if hasattr(self, "log_f") and not self.output_f.closed:
            self.output_f.close()


class LogStatisticsCalculator(BaseCalculator):
    r""" Statistics from logs
    """

    def _mean_and_stdev(self, data):
        mean = stat.mean(data)
        stdev = stat.stdev(data)
        return (mean, stdev)

    def _cal_stat(self):
        accuracies = list()
        precisions = list()
        recalls = list()
        f1_scores = list()
        auc_scores = list()
        for eva in self.evaluators:
            best_index = np.argmax(eva.accuracy_scores())
            accuracies.append(eva.accuracy_scores()[best_index])
            prec, rec, f1 = eva.precision_recall_fbeta_scores()
            precisions.append(prec[best_index])
            recalls.append(rec[best_index])
            f1_scores.append(f1[best_index])
            auc_scores.append(eva.roc_auc()[best_index])
        acc_stat = self._mean_and_stdev(accuracies)
        # convert accuracy to percentage representation
        acc_stat = [num * 100 for num in acc_stat]
        prec_stat = self._mean_and_stdev(precisions)
        rec_stat = self._mean_and_stdev(recalls)
        f1_stat = self._mean_and_stdev(f1_scores)
        auc_stat = self._mean_and_stdev(auc_scores)
        return [acc_stat, prec_stat, rec_stat, f1_stat, auc_stat]

    @property
    def statistics_(self):
        try:
            return self._statistics_
        except AttributeError:
            self._statistics_ = self._cal_stat()
            return self._statistics_

    def statistics_str(self):
        stat_str = "Accuracy\tPrecision\tRecall\tF1\tAUC\n"
        for stats in self.statistics_:
            stat_str += "{:.2f} ± {:.2f}\t".format(stats[0], stats[1])
        print(stat_str)
        self.output_f.write(stat_str + "\n")
