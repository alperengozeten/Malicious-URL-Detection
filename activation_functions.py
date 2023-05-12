from abc import abstractmethod, ABC
from typing import Any
import numpy as np

class ActivationFunction(ABC):

    # V denotes the local induced field
    # return Z, which is the output of the activation function
    @abstractmethod
    def __call__(self, V):
        pass

    # dZ denotes the derivative of the layer output
    # V denotes the local induced field at the layer
    # returns the gradient of the cost w.r.t V
    @abstractmethod
    def derivative(self, V, dZ):
        pass

    def __str__(self) -> str:
        return str(self.__class__)
    
    def __repr__(self) -> str:
        return str(self.__class__)
    
class ReLU(ActivationFunction):
    # The ReLU (Rectified Linear Unit) activation
    # function implementation.
    # f(x) = max(0, x)

    def __call__(self, V):
        return np.maximum(V, 0)
    
    def derivative(self, V, dZ):
        return np.where(V <= 0, 0, dZ)
    
class Sigmoid(ActivationFunction):
    # The Linear activation
    # function implementation.
    # f(x) = 1 / (1 + e^-x)

    def __call__(self, V):
        return 1 / (1 + np.exp(-V))
    
    def derivative(self, V, dZ):
        return dZ * (np.exp(-V) / ((1 + np.exp(-V)) ** 2))

class Linear(ActivationFunction):
    # The Linear activation
    # function implementation.
    # f(x) = x

    def __call__(self, V):
        return V
    
    def derivative(self, V, dZ):
        return dZ