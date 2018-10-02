import sys
sys.path.append('./utils/function')
import Modelling


from abc import ABCMeta, abstractmethod

class Modelling(ABCMeta):
    def __init__(verbose = False):
        self.verbose = verbose

    @abstractmethod
    def forward(self, df, cols):
        pass

    def __call__(self):
        self.forward()
