# Base class of the dataloaders
import abc

class _BaseDataLoader:

    def __init__(self, path):
        self.path = path
    
    @abc.abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_unlabeled(self):
        pass
