import argparse


class TrainingArgs(argparse.ArgumentParser):

    def __init__(self):
        super(TrainingArgs, self).__init__()
        self.add_argument("-p", "--data-path")
        self.add_argument("-b", "--batch-size", type=int)
        self.add_argument("-l", "--learning-rate", type=float)
        self.add_argument("-e", "--epochs", type=int)
        self.add_argument("-r", "--if-hard", action="store_true")
        self.add_argument("--es-patience", type=int, default=3,
                          help="early_stop callback patience.")
        self.add_argument("--log-path", type=str, default="../logs",
                          help="path to the log directory")
        self.add_argument("-m", "--comment", type=str, default=None,
                          help="comment for the training")
        self.add_argument("--unlabeled-weight", type=float, default=0.5,
                          help="weight of unlabeled data in loss function")
