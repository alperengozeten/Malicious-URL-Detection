import numpy as np
from activation_functions import ActivationFunction      
from typing import Any, Type, Tuple, Iterable
from collections import defaultdict, namedtuple
from tqdm import tqdm
from metrics import mse, mae, mape, r2

def relu(z):
    return np.maximum(0, z)


def relu_backward(z):
    return np.where(z > 0, 1, 0)

# Weight and gradient containers for readability
RecurrentLayerWeights = namedtuple('RecurrentLayerWeights', ['b', 'Wx', 'Wh'])
RecurrentLayerGradients = RecurrentLayerWeights

FullyConnectedLayerWeights = namedtuple('FullyConnectedLayerWeights', ['b', 'W'])
FullyConnectedLayerGradients = FullyConnectedLayerWeights

DEFAULT_METRICS = {
    'MSE': mse,
    'MAE': mae,
    'MAPE': mape,
    'R2': r2,
}

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

'''
class NeuralNetwork:

    def __init__(self, layers : Iterable[Tuple[int, int, Type[ActivationFunction]]]) -> None:
        self._layers = [FullyConnectedLayer(in_features, out_features, activation_func) 
                        for in_features, out_features, activation_func in layers]
    
    @property
    def layers(self):
        return self._layers'''

class NeuralNetwork:
    def __init__(self, n_neurons):
        self.input_units = None                      # will be determined when while training
        self.n_neurons = n_neurons                   # number of neurons in the layers of MLP
        self.perceptron = None                       # MLP layers

    def initialize_layers(self):
        """Initializes the weights and the hidden state of the recurrent layer"""
        rng = np.random.default_rng()

        # initialize MLP weights
        self.perceptron = []
        for i, n in enumerate(self.n_neurons):
            n_prev = self.n_neurons[i - 1] if i != 0 else self.input_units
            bound = np.sqrt(6 / (n + n_prev))
            b = rng.uniform(-bound, bound, size=(n, 1))
            W = rng.uniform(-bound, bound, size=(n, n_prev))

            layer = FullyConnectedLayerWeights(b, W)
            self.perceptron.append(layer)

    def __call__(self, X):
        """Inference mode forward pass through the network"""
        Z = X.T
        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)
        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = V
        return Z.T

    def forward(self, X, y):
        """
        Training mode forward pass though the network.
        Caches the pre- and post-activation outputs of the layers
        """
        Z = X
        V_cache, Z_cache = [], []
        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)

            V_cache.append(V)
            Z_cache.append(Z)

        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = V

        V_cache.append(V)
        Z_cache.append(Z)

        # compute training metrics
        J = {metric: fn(y.T, Z.T) for metric, fn in DEFAULT_METRICS.items()}
        return J, V_cache, Z_cache

    def backward(self, X, y, V_cache, Z_cache):
        """Training mode backward pass through the network"""
        # calculate the gradients for the MLP
        gradients = []

        delta = (2 / Z_cache[-1].shape[1]) * np.subtract(Z_cache[-1], y) # np.sign(Z_cache[-1] - y)
        db = np.mean(delta, axis=1, keepdims=True)
        dW = delta @ Z_cache[-2].T / Z_cache[-2].shape[-1]
        gradients.append(FullyConnectedLayerGradients(db, dW))

        for i in reversed(range(1, len(self.perceptron) - 1)):
            delta = (self.perceptron[i + 1].W.T @ delta) * relu_backward(V_cache[i])
            db = np.mean(delta, axis=1, keepdims=True)
            dW = delta @ Z_cache[i - 1].T / Z_cache[i - 1].shape[-1]
            gradients.append(FullyConnectedLayerGradients(db, dW))

        delta = (self.perceptron[1].W.T @ delta) * relu_backward(V_cache[0])
        db = np.mean(delta, axis=1, keepdims=True)
        dW = (delta @ X.T) / X.shape[-1]
        gradients.append(FullyConnectedLayerGradients(db, dW))

        gradients.reverse()  # reverse the gradient list to get the correct order
        return gradients

    def step(self, X, y):
        """Combines a forward and a backward pass through the network"""
        J, V_cache, Z_cache = self.forward(X, y)
        gradients = self.backward(X, y, V_cache, Z_cache)
        return J, gradients

    def fit(self,
            X,
            y,
            X_valid,
            y_valid,
            alpha=0.1,
            momentum=0.85,
            epochs=50,
            batch_size=32,
            patience=5,
            min_delta=1e-7,
            shuffle=True,
            cold_start=False
            ):
        """
        Train the neural network
        :param X: the training features
        :param y: the training labels
        :param alpha: the learning rate
        :param momentum: the momentum
        :param epochs: the number of training epochs
        :param batch_size: the size of the training batches
        :param unfold: number of time steps to backpropagate in time
        :param shuffle: whether to shuffle the data each epoch
        :param X_valid: the validation features
        :param y_valid: the validation labels
        :param cold_start: whether to reinitialize the weights before training
        :return:
        """
        X = np.asarray(X)
        y = np.asarray(y)
        X_valid = np.asarray(X_valid)
        y_valid = np.asarray(y_valid)
        
        if cold_start or self.perceptron is None:
            self.input_units = X.shape[-1]
            self.initialize_layers()

        n_batches = len(X) // batch_size
        history = defaultdict(list)

        delta_weights_prev = None
        n_no_improvement = 0
        for epoch in (progress_bar := tqdm(range(epochs))):
            # obtain the shuffled indices for training and validation
            train_indices = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
            batch_avg_losses = []
            for batch in range(n_batches):
                # get the batch data using the shuffled indices
                batch_indices = train_indices[batch * batch_size: (batch + 1) * batch_size]
                X_batch = X[batch_indices].T
                y_batch = y[batch_indices].T

                # forward and backward passes through the network
                J, gradients = self.step(X_batch, y_batch)

                batch_avg_losses.append(J)

                # calculate the updates given previous updates and gradients
                delta_weights = self.calculate_updates(delta_weights_prev, gradients, momentum)
                self.apply_updates(delta_weights, alpha)

                # save previous updates for momentum
                delta_weights_prev = delta_weights

            # test model performance on validation dataset
            valid_avg_losses = {metric: np.mean(fn(y_valid, self(X_valid)))
                                for metric, fn in DEFAULT_METRICS.items()}
            train_avg_losses = {metric: np.mean([loss[metric] for loss in batch_avg_losses])
                                for metric in DEFAULT_METRICS.keys()}

            # log training and validation metrics

            for metric in DEFAULT_METRICS.keys():
                history[f'train_{metric}'].append(train_avg_losses[metric])
                history[f'valid_{metric}'].append(valid_avg_losses[metric])

            progress_bar.set_description_str(f'n_neurons={"-".join([str(i) for i in self.n_neurons])}, '
                                             f'alpha={alpha}, momentum={momentum}, batch_size={batch_size}')
            progress_bar.set_postfix_str(f'train_mse={train_avg_losses["MSE"]:.7f}, '
                                         f'valid_mse={valid_avg_losses["MSE"]:.7f}')

            # stop the training if there is no improvement in last "tolerance" episodes
            if epoch > 2 and history['valid_MSE'][-2] - history['valid_MSE'][-1] < min_delta:
                n_no_improvement += 1
                if n_no_improvement > patience:
                    break
            else:
                n_no_improvement = 0
        return history

    @staticmethod
    def calculate_updates(delta_weights_prev, gradients, momentum):
        """Calculate the weight updates with momentum given previous updates and gradients"""
        delta_weights = [
            FullyConnectedLayerWeights(*[momentum * delta_w_prev + (1 - momentum) * grads
                                         for delta_w_prev, grads in zip(delta_weights_prev_, gradients_)])
            for delta_weights_prev_, gradients_ in zip(delta_weights_prev, gradients)
        ] if delta_weights_prev else gradients
        return delta_weights

    def apply_updates(self, delta_weights, alpha):
        """Apply the calculated updates to the network weights"""
        self.perceptron = [
            FullyConnectedLayerWeights(*[w - alpha * delta for w, delta in zip(layer, delta_weights_)])
            for layer, delta_weights_ in zip(self.perceptron, delta_weights)
        ]