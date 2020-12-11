import argparse


class BaseArgs(argparse.ArgumentParser):
    def __init__(self):
        super(BaseArgs, self).__init__()
        self.add_argument("-p", "--data-path", help="Path to the training dataset.")
        self.add_argument("-b", "--batch-size", type=int, default=32)
        self.add_argument("-e", "--epochs", type=int, default=100)
        self.add_argument(
            "--es-patience", type=int, default=3, help="early_stop callback patience."
        )
        self.add_argument(
            "--log-path", type=str, default="../logs", help="path to the log directory"
        )


class TrainingArgs(BaseArgs):
    def __init__(self):
        super(TrainingArgs, self).__init__()
        self.add_argument("-l", "--learning-rate", type=float, default=0.001)
        self.add_argument(
            "-r",
            "--if-hard",
            action="store_true",
            help="If to use hard label when applying Noisy " "Student method",
        )
        self.add_argument(
            "-m", "--comment", type=str, default=None, help="comment for the training"
        )
        self.add_argument(
            "--unlabeled-weight",
            type=float,
            default=0.5,
            help="weight of unlabeled data in loss function",
        )


class LinearModelTrainingArgs(BaseArgs):
    def __init__(self):
        super(LinearModelTrainingArgs, self).__init__()
        self.add_argument(
            "--repeat",
            type=int,
            default=3,
            help="Times to repeat when training the model with " "Noisy Student.",
        )
        self.add_argument(
            "--rand-seed",
            type=int,
            default=None,
            help="Random seed used to generate training and " "testing data.",
        )
        self.add_argument(
            "--learning-rate",
            type=float,
            default=1e-6,
            help="The initial learning rate.",
        )
        self.add_argument(
            "--drop-rate",
            type=float,
            default=0.3,
            help="The drop rate of the dropout layers."
        )


class LMOutsideDataArgs(LinearModelTrainingArgs):
    def __init__(self):
        super(LMOutsideDataArgs, self).__init__()
        self.add_argument("--outside-path", help="Path to the outside dataset.")


class LMMixupArgs(LinearModelTrainingArgs):
    def __init__(self):
        super(LMMixupArgs, self).__init__()
        self.add_argument(
            "--mixup",
            default=None,
            type=float,
            help="The coefficient of the Beta distribution " "in mixup.",
        )
        self.add_argument(
            "--mixup-repeat",
            default=None,
            type=int,
            help="Times to repeat the dataset when apply mixup.",
        )


class LMMixupOutsideDataArgs(LMOutsideDataArgs, LMMixupArgs):
    def __init__(self):
        super(LMMixupOutsideDataArgs, self).__init__()


class ConventionalArgs(LinearModelTrainingArgs):
    def __init__(self):
        super(ConventionalArgs, self).__init__()
        self.add_argument(
            "--n-estimators", default=500, type=int, help="Number of estimators."
        )
        self.add_argument(
            "--max-depth", default=None, type=int, help="The max depth of the trees."
        )
