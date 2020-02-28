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
        """ Readout the predictions from the log file.
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
