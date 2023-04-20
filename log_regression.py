import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Literal, Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("data/url_processed.csv")

# shuffle the data frame
df = df.sample(frac = 1)

X = df[['is_ip', 'url_len', 'subdomain_len', 'tld_len', 'fld_len', 'url_path_len',
       'url_alphas', 'url_digits', 'url_puncs', 'count.', 'count@', 'count-',
       'count%', 'count?', 'count=',
       'pc_alphas', 'pc_digits', 'pc_puncs', 'count_dirs',
       'contains_shortener', 'first_dir_len',
       'url_len_q', 'fld_len_q']]

print(X.head())

y = df['binary_label']

print(X.isna().sum())

# fill the nan values with 0
X = X.fillna(0)

# check again the number of nan values
print(X.isna().sum())

# divide the data into train, test and validation datasets
n = len(y)

train_size = int(n * 0.7)
valid_size = int(n * 0.1)
test_size = int(n * 0.2)

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_valid = X.iloc[train_size: train_size + valid_size]
y_valid = y.iloc[train_size: train_size + valid_size]

X_test = X.iloc[train_size + valid_size:]
y_test = y.iloc[train_size + valid_size:]

print(X_train.shape, X_valid.shape, X_test.shape)

print(X_train.head())

## Implement the Logistic Regression model

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

class LogisticRegression:
    """
    Logistic regression model that fits the parameters using gradient descent.

    Attributes:
        b : np.ndarray
            the bias vector of the model
        W : np.ndarray
            the weight matrix of the model
        alpha : float
            the learning rate of the model
    """
    def __init__(self, initializer: Literal['normal', 'uniform', 'zeros']):
        """
        The init method of the LogisticRegression model
        :param initializer: the weight initialization method
        """
        self._b = None
        self._W = None
        self.initializer = initializer
        self.check_initializer()

    @property
    def b(self) -> np.ndarray:
        """
        The y-intercept of the model
        :return: the y-intercept
        """
        return self._b

    @property
    def W(self) -> np.ndarray:
        """
        The weight matrix of the model
        :return: the weight matrix
        """
        return self._W

    def __repr__(self) -> str:
        """
        Returns the initialization signature of the instance
        :return: the string representation
        """
        return f'LogisticRegression(initializer={self.initializer})'

    def __str__(self) -> str:
        """
        Calls the repr method of the class
        :return: the string representation
        """
        return repr(self)
    
    def check_initializer(self):
        """
        Checks whether an initializer is implemented
        """
        return self.initializer in {'normal', 'uniform', 'zeros'}
    
    def initialize_parameters(self,
                              in_features: int,
                              initializer: Literal['normal', 'uniform', 'zeros'] = None):
        """
        Initializes the model parameters from a standart normal distribution
        
        :param in_features: the number of features
        """
        if initializer is None:
            initializer = self.initializer
        if initializer == 'zeros':
            self._b = 0
            self._W = np.zeros(in_features)
        else:
            rng = np.random.default_rng()
            if initializer == 'normal':
                self._b = rng.normal(0, 1, size=1)
                self._W = rng.normal(0, 1, size=in_features)
            elif initializer == 'uniform':
                self._b = rng.uniform(-0.01, 0.01, size=1)
                self._W = rng.uniform(-0.01, 0.01, size=in_features)
            else:
                raise NotImplementedError('Only "normal", "uniform" and '
                                          '"zeros" are supported as initializer.')

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the probability of being in the positive class
        :param X: the feature matrix
        :return: predictions
        """
        if self.b is None or self.W is None:
            raise RuntimeError('The model is not fit.')
        return sigmoid(self.b + X @ self.W)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_valid: np.ndarray,
            y_valid: np.ndarray,
            epochs: int = 1,
            batch_size: int = None,
            learning_rate: float = 0.01,
            shuffle: bool = True) -> np.ndarray:
        """
        Calculates the weights and bias of the model using the gradient descent algorithm
        :param X: the feature matrix
        :param y: the target vector
        :param epochs: the number of iterations over the training set
        :param batch_size: the batch size. Set to None to set batch size
                           equal to the train dataset size
        :param X_valid: the feature matrix of the validation dataset
        :param y_valid: the target vector of the validation dataset
        :return: the accuracy history
        """
        X = np.asarray(X)
        y = np.asarray(y)
        X_valid = np.asarray(X_valid)
        y_valid = np.asarray(y_valid)
        
        if self.b is None or self.W is None:
            self.initialize_parameters(X.shape[-1])

        if batch_size is None:
            batch_size = len(y)
        n_batches = len(y) // batch_size
        
        accuracy_ = accuracy(y_valid, self.predict(X_valid))
        history = [accuracy_]
        for epoch in tqdm(range(epochs)):
            indices = np.random.permutation(len(y_train)) if shuffle else np.arange(len(y_train))
            for batch in range(n_batches):
                batch_indices = indices[batch * batch_size: (batch + 1) * batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                grad_b, grad_W = self._calculate_gradients(X_batch, y_batch)
                self._b -= learning_rate * grad_b
                self._W -= learning_rate * grad_W
            accuracy_ = accuracy(y_valid, self.predict(X_valid))
            history.append(accuracy_)
        return np.asarray(history)

    def predict(self, 
                X: np.ndarray,
                threshold: float = 0.5) -> np.ndarray:
        """
        Predicts the class labels of the inputs
        
        :param X: the input data
        :param threshold: the threshold over which the class will be considered positive
        """
        return np.asarray(self(X) > threshold, dtype=np.int32)

    def _calculate_gradients(self,
                             X: np.ndarray,
                             y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the gradients for binary crossentropy loss
        with current bias and weights
        :param X: the feature matrix
        :param y: the target vector
        :return: the gradients of the bias and the weights
        """
        y_pred = self(X)
        grad_b = np.mean(y_pred - y)
        grad_W = X.T @ (y_pred - y) / len(y)
        return grad_b, grad_W

"""
model_sgd = LogisticRegression(initializer='normal')
history_sgd = model_sgd.fit(X_train, y_train,
                            X_valid, y_valid,
                            epochs=500,
                            batch_size=1,
                            learning_rate=1e-3)

"""

### Train the logistic regression model with batch size = 64

model_mini_batch = LogisticRegression(initializer='normal')
history_batch_32 = model_mini_batch.fit(X_train, y_train,
                                          X_valid, y_valid,
                                          epochs=100,
                                          batch_size=32,
                                          learning_rate=1e-3)

history_batch_64 = model_mini_batch.fit(X_train, y_train,
                                          X_valid, y_valid,
                                          epochs=100,
                                          batch_size=64,
                                          learning_rate=1e-3)


### Train the logistic regression model with batch size = dataset size

model_full_batch = LogisticRegression(initializer='normal')
history_batch_128 = model_full_batch.fit(X_train, y_train,
                                          X_valid, y_valid,
                                          epochs=100,
                                          batch_size=128,
                                          learning_rate=1e-3)

model_full_batch = LogisticRegression(initializer='normal')
history_batch_256 = model_full_batch.fit(X_train, y_train,
                                          X_valid, y_valid,
                                          epochs=100,
                                          batch_size=256,
                                          learning_rate=1e-3)

x = list(range(100 + 1))

plt.figure(figsize=(18, 12))
plt.title('Batch Size Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
#plt.plot(x, history_sgd, label='Batch size = 1')
plt.plot(x, history_batch_32, label='Batch size = 32')
plt.plot(x, history_batch_64, label='Batch size = 64')
plt.plot(x, history_batch_128, label='Batch size = 128')
plt.plot(x, history_batch_256, label='Batch size = 256')
plt.legend()
plt.show()