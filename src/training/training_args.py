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