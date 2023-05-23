import numpy as np
from activation_functions import ActivationFunction      
from typing import Any, Type, Tuple, Iterable
from collections import defaultdict, namedtuple
from tqdm import tqdm
from metrics import mse, mae, r2

def relu(z):
    return np.maximum(0, z)

def relu_backward(z):
    return np.where(z > 0, 1, 0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_backward(z):
    return (1 - sigmoid(z)) * sigmoid(z)

def calc_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()
    return np.mean(y_true == y_pred)

FullyConnectedLayerWeights = namedtuple('FullyConnectedLayerWeights', ['b', 'W'])
FullyConnectedLayerGradients = FullyConnectedLayerWeights

DEFAULT_METRICS = {
    'MSE': mse,
    'MAE': mae,
    'R2': r2,
}

class NeuralNetwork:
    def __init__(self, n_neurons):
        self.input_units = None
        # number of neurons in the layer    
        self.n_neurons = n_neurons       
        self.perceptron = None 

    def init_network(self):
        rng = np.random.default_rng()

        # init weights
        self.perceptron = []
        for i, n in enumerate(self.n_neurons):
            n_prev = self.n_neurons[i - 1] if i != 0 else self.input_units
            bound = np.sqrt(6 / (n + n_prev))
            b = rng.uniform(-bound, bound, size=(n, 1))
            W = rng.uniform(-bound, bound, size=(n, n_prev))

            layer = FullyConnectedLayerWeights(b, W)
            self.perceptron.append(layer)

    def __call__(self, X):
        Z = X.T
        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)
        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = sigmoid(V)
        return Z.T
    
    # predict the output with respect to some threshold
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.asarray(self(X) > threshold, dtype=np.int32)

    # forward pass through the layers by caching the outputs
    def forward(self, X, y):
        Z = X
        V_cache, Z_cache = [], []
        for layer in self.perceptron[:-1]:
            V = layer.W @ Z + layer.b
            Z = relu(V)

            V_cache.append(V)
            Z_cache.append(Z)

        V = self.perceptron[-1].W @ Z + self.perceptron[-1].b
        Z = sigmoid(V)

        V_cache.append(V)
        Z_cache.append(Z)

        # compute metrics
        J = {metric: fn(y.T, Z.T) for metric, fn in DEFAULT_METRICS.items()}
        return J, V_cache, Z_cache

    def backward(self, X, y, V_cache, Z_cache):
        
        # gradients for the model
        gradients = []

        delta = (1 / Z_cache[-1].shape[1]) * np.subtract(Z_cache[-1], y)
        db = np.mean(delta, axis=1, keepdims=True)
        dW = delta @ Z_cache[-2].T / Z_cache[-2].shape[-1]
        gradients.append(FullyConnectedLayerGradients(db, dW))

        # calculate gradients for the hidden layer
        for i in reversed(range(1, len(self.perceptron) - 1)):
            delta = (self.perceptron[i + 1].W.T @ delta) * relu_backward(V_cache[i])
            db = np.mean(delta, axis=1, keepdims=True)
            dW = delta @ Z_cache[i - 1].T / Z_cache[i - 1].shape[-1]
            gradients.append(FullyConnectedLayerGradients(db, dW))

        # treat the first hidden layer differently by using X (design matrix)
        delta = (self.perceptron[1].W.T @ delta) * relu_backward(V_cache[0])
        db = np.mean(delta, axis=1, keepdims=True)
        dW = (delta @ X.T) / X.shape[-1]
        gradients.append(FullyConnectedLayerGradients(db, dW))

        # get the correct order
        gradients.reverse() 
        return gradients

    # forward and backward pass combined
    def step(self, X, y):
        J, V_cache, Z_cache = self.forward(X, y)
        gradients = self.backward(X, y, V_cache, Z_cache)
        return J, gradients

    def fit(self,
            X, # training features
            y, # training labels
            X_valid, # validation features
            y_valid, # validation labels
            alpha=0.1, # learning rate
            momentum=0.85, # momentum
            epochs=50, # number of training iterations
            batch_size=32, # batch size for training
            patience=5, # patience
            min_delta=1e-7, # patience criteria
            shuffle=True,
            cold_start=False
            ):

        X = np.asarray(X)
        y = np.asarray(y)
        X_valid = np.asarray(X_valid)
        y_valid = np.asarray(y_valid)
        
        if cold_start or self.perceptron is None:
            self.input_units = X.shape[-1]
            self.init_network()

        n_batches = len(X) // batch_size
        history = defaultdict(list)

        delta_weights_prev = None
        n_no_improvement = 0
        for epoch in (progress_bar := tqdm(range(epochs))):
            # get the shuffled data
            train_indices = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
            batch_avg_losses = []
            for batch in range(n_batches):
                # get the batch data from the shuffle
                batch_indices = train_indices[batch * batch_size: (batch + 1) * batch_size]
                X_batch = X[batch_indices].T
                y_batch = y[batch_indices].T

                # take one step: forward and backward combined
                J, gradients = self.step(X_batch, y_batch)

                batch_avg_losses.append(J)

                # update weights
                delta_weights = self.add_velocity(delta_weights_prev, gradients, momentum)
                self.update_params(delta_weights, alpha)

                # updates saved for momentum
                delta_weights_prev = delta_weights

            # evaluate the model on validation dataset
            valid_avg_losses = {metric: np.mean(fn(y_valid, self(X_valid)))
                                for metric, fn in DEFAULT_METRICS.items()}
            train_avg_losses = {metric: np.mean([loss[metric] for loss in batch_avg_losses])
                                for metric in DEFAULT_METRICS.keys()}

            valid_acc_log = calc_accuracy(y_valid, self.predict(X_valid))
            train_acc_log = calc_accuracy(y, self.predict(X))

            history['train_acc'].append(train_acc_log)
            history['valid_acc'].append(valid_acc_log)

            for metric in DEFAULT_METRICS.keys():
                history[f'train_{metric}'].append(train_avg_losses[metric])
                history[f'valid_{metric}'].append(valid_avg_losses[metric])

            # for output display
            progress_bar.set_description_str(f'n_neurons={"-".join([str(i) for i in self.n_neurons])}, '
                                             f'alpha={alpha}, momentum={momentum}, batch_size={batch_size}')
            progress_bar.set_postfix_str(f'train_acc={train_acc_log:.7f}, '
                                         f'valid_acc={valid_acc_log:.7f}')

            # exit training if there is no significant improvement in the model
            if epoch > 2 and history['valid_MSE'][-2] - history['valid_MSE'][-1] < min_delta:
                n_no_improvement += 1
                if n_no_improvement > patience:
                    break
            else:
                n_no_improvement = 0
        return history

    # calculate the updates by introducing momentum
    @staticmethod
    def add_velocity(delta_weights_prev, gradients, momentum):
        delta_weights = [
            FullyConnectedLayerWeights(*[momentum * delta_w_prev + (1 - momentum) * grads
                                         for delta_w_prev, grads in zip(delta_weights_prev_, gradients_)])
            for delta_weights_prev_, gradients_ in zip(delta_weights_prev, gradients)
        ] if delta_weights_prev else gradients
        return delta_weights

    # apply updates on to the weights
    def update_params(self, delta_weights, alpha):
        self.perceptron = [
            FullyConnectedLayerWeights(*[w - alpha * delta for w, delta in zip(layer, delta_weights_)])
            for layer, delta_weights_ in zip(self.perceptron, delta_weights)
        ]