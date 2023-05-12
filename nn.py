import numpy as np
from activation_functions import ActivationFunction      
from typing import Any, Type, Tuple, Iterable

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
    
    def __call__(self, X):
        return self.activation(self.V(X))
    
    # returns the induced local field at this layer
    def V(self, X):
        return X @ self._W + self._b
    
    def init_params(self):
        generator = np.random.default_rng()
        self._W = generator.uniform(-1, 1, (self._in_features, self._out_features))
        self._b = generator.uniform(-1, 1, (1, self._out_features))
    
    def update_params(self, rate, dW, db):
        self._W = self._W - dW * rate
        self._b = self._b - db * rate

    def __repr__(self) -> str:
        return 'FullyConnectedLayer(in_features=' + str(self._in_features) + ', out_features=' + str(self._out_features) + ', activation:' + str(self._activation_func)

class NeuralNetwork:

    def __init__(self, layers : Iterable[Tuple[int, int, Type[ActivationFunction]]]) -> None:
        self._layers = [FullyConnectedLayer(in_features, out_features, activation_func) 
                        for in_features, out_features, activation_func in layers]
    
    @property
    def layers(self):
        return self._layers