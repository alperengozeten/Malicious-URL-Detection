import numpy as np
from activation_functions import ActivationFunction
from typing import Type

class FullyConnectedLayer:

    def __init__(self, in_features, out_features, activation_func : Type[ActivationFunction]) -> None:
        self._activation_func = activation_func()
        self._in_features = in_features
        self._out_features = out_features
        self._W = None
        self._b = None
        self.init_params()
    
    @property
    def W(self):
        return self._W
    
    @property
    def b(self):
        return self._b
    
    @property
    def activation(self):
        return self._activation_func
    
    @property
    def in_features(self):
        return self._in_features

    @property
    def out_features(self):
        return self._out_features