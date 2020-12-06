from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    def __init__(self, params, task_type=None):
        self.params = params
        self.model = None
        self.task_type = task_type

    @abstractmethod
    def fit(self, tr_x, tr_y, te_x, va_x=None, va_y=None, cat_features=None):
        pass

    @abstractmethod
    def predict(self, te_x, cat_features=None):
        pass
