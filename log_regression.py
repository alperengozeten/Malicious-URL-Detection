import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Literal, Tuple
from tqdm import tqdm

df = pd.read_csv("data/url_processed.csv")
np.random.seed(2023)

# shuffle the data frame
df = df.sample(frac = 1)

# our feature matrix
X = df[['use_of_ip', 'url_length', 'subdomain_length', 'tld_length', 'fld_length', 'path_length',
       'count_letters', 'count_digits', 'count_puncs', 'count.', 'count@', 'count-',
       'count%', 'count?', 'count=',
       'letters_ratio', 'digit_ratio', 'punc_ratio', 'count_dirs',
       'contains_shortener', 'first_dir_length',
       'url_length_q', 'fld_length_q', 'https', 'count-https', 'count-http', 'count-www']]

print(X.head())

# labels
y = df['is_malicious']

print(X.isna().sum())

# fill the nan values with 0
X = X.fillna(0)

# check again the number of nan values
print(X.isna().sum())

# divide the data into train, test and validation datasets
length = len(y)

# %70 train, %10 validation, %20 test
train_set_size = int(length * 0.7)
valid_set_size = int(length * 0.1)
test_set_size = int(length * 0.2)


X_train = X.iloc[:train_set_size]
y_train = y.iloc[:train_set_size]

X_valid = X.iloc[train_set_size: train_set_size + valid_set_size]
y_valid = y.iloc[train_set_size: train_set_size + valid_set_size]

X_test = X.iloc[train_set_size + valid_set_size:]
y_test = y.iloc[train_set_size + valid_set_size:]

print(X_train.shape, X_valid.shape, X_test.shape)

print(X_train.head())

## Implement the Logistic Regression model

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calc_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)

# logistic regression class
class LogisticRegression:

    # param b : y interception
    # param W : weights
    # initializer : initial distribution of weights
    def __init__(self, initializer: Literal['normal', 'uniform', 'zeros']):

        self._b = None
        self._W = None
        self.initializer = initializer

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def W(self) -> np.ndarray:
        return self._W

    def __repr__(self) -> str:
        return f'LogisticRegression(initializer={self.initializer})'

    def __str__(self) -> str:
        # calls __repr__
        return repr(self)
    
    # initializes the parameters according to given features
    def init_params(self, in_features: int,
                              initializer: Literal['normal', 'uniform', 'zeros'] = None):
        if initializer is None:
            initializer = self.initializer
        if initializer == 'zeros':
            self._b = 0
            self._W = np.zeros(in_features)
        else:
            random = np.random.default_rng()
            if initializer == 'uniform':
                self._b = random.uniform(-0.01, 0.01, size=1)
                self._W = random.uniform(-0.01, 0.01, size=in_features)
            elif initializer == 'normal':
                self._b = random.normal(0, 1, size=1)
                self._W = random.normal(0, 1, size=in_features)


    # calculates the probability for the output of the sigmoid function
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.b + X @ self.W)

    # fits the model
    def fit(self, X: np.ndarray, y: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray, epochs: int = 1,
            batch_size: int = None, learning_rate: float = 0.01, shuffle: bool = True) -> np.ndarray:

        # X : design matrix
        # y : label vector
        # X_valid : validation set design matrix
        # y_valid : validation set label vector
        X = np.asarray(X)
        y = np.asarray(y)
        X_valid = np.asarray(X_valid)
        y_valid = np.asarray(y_valid)
        
        if self.b is None or self.W is None:
            self.init_params(X.shape[-1])

        if batch_size is None:
            batch_size = len(y)
        n_batches = len(y) // batch_size
        
        acc_log = calc_accuracy(y_valid, self.predict(X_valid))
        history = [acc_log]
        for _ in tqdm(range(epochs)):
            idx = np.random.permutation(len(y)) if shuffle else np.arange(len(y))
            for batch in range(n_batches):
                batch_idx = idx[batch * batch_size: (batch + 1) * batch_size]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                grad_b, grad_W = self._calc_gradients(X_batch, y_batch)
                self._b = self._b - learning_rate * grad_b
                self._W = self._W - learning_rate * grad_W
            acc_log = calc_accuracy(y_valid, self.predict(X_valid))
            history.append(acc_log)

        return np.asarray(history)

    # predict the output with respect to some threshold
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.asarray(self(X) > threshold, dtype=np.int32)

    # calculate gradients and update later
    def _calc_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self(X)
        grad_b = np.mean(y_pred - y)
        grad_W = X.T @ (y_pred - y) / len(y)
        return grad_b, grad_W

### Train the logistic regression model with changing batch sizes

acc_batch_normal = []
acc_batch_uniform = []
acc_batch_zeros = []
def logistic_reg_trainer(initializer, batch_size):
    model = LogisticRegression(initializer)
    acc = model.fit(X_train, y_train,
                    X_valid, y_valid,
                    epochs=20,
                    batch_size=batch_size,
                    learning_rate=1e-3)
    if initializer == 'normal' : acc_batch_normal.append(acc)
    if initializer == 'uniform' : acc_batch_uniform.append(acc)
    if initializer == 'zeros' : acc_batch_zeros.append(acc)

fig, axs = plt.subplots(2, 2, figsize=(18, 12))
axs[1, 1].remove()

# weights initialized NORMAL dist with different batches
logistic_reg_trainer('normal', 32)
logistic_reg_trainer('normal', 64)
logistic_reg_trainer('normal', 128)
logistic_reg_trainer('normal', 256) 
# plot normal
x = list(range(20 + 1))
#plt.figure(figsize=(18, 12))
axs[0,0].set_title('Batch Size Comparison for Normal initializer')
axs[0,0].set_xlabel('Epochs')
axs[0,0].set_ylabel('Validation accuracy')
axs[0,0].plot(x, acc_batch_normal[0], label='Batch size = 32')
axs[0,0].plot(x, acc_batch_normal[1], label='Batch size = 64')
axs[0,0].plot(x, acc_batch_normal[2], label='Batch size = 128')
axs[0,0].plot(x, acc_batch_normal[3], label='Batch size = 256')
axs[0,0].legend()

# weights initialized UNIFORM dist with different batches
logistic_reg_trainer('uniform', 32)
logistic_reg_trainer('uniform', 64)
logistic_reg_trainer('uniform', 128)
logistic_reg_trainer('uniform', 256)
# plot uniform
#plt.figure(figsize=(18, 12))
axs[0,1].set_title('Batch Size Comparison for Uniform initializer')
axs[0,1].set_xlabel('Epochs')
axs[0,1].set_ylabel('Validation accuracy')
axs[0,1].plot(x, acc_batch_uniform[0], label='Batch size = 32')
axs[0,1].plot(x, acc_batch_uniform[1], label='Batch size = 64')
axs[0,1].plot(x, acc_batch_uniform[2], label='Batch size = 128')
axs[0,1].plot(x, acc_batch_uniform[3], label='Batch size = 256')
axs[0,1].legend()

# weights initialized ZEROS with different batches
logistic_reg_trainer('zeros', 32)
logistic_reg_trainer('zeros', 64)
logistic_reg_trainer('zeros', 128)
logistic_reg_trainer('zeros', 256)
# plot zeros
#plt.figure(figsize=(18, 12))
axs[1, 0].set_title('Batch Size Comparison for Zeros initializer')
axs[1, 0].set_xlabel('Epochs')
axs[1, 0].set_ylabel('Validation accuracy')
axs[1, 0].plot(x, acc_batch_zeros[0], label='Batch size = 32')
axs[1, 0].plot(x, acc_batch_zeros[1], label='Batch size = 64')
axs[1, 0].plot(x, acc_batch_zeros[2], label='Batch size = 128')
axs[1, 0].plot(x, acc_batch_zeros[3], label='Batch size = 256')
axs[1, 0].legend()

plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

for i in range(1, 5):
    print('Final Validation accuracy for Normal Initialization With Batch Size: %d is : %f' % (32 * i, acc_batch_normal[i - 1][-1]))
for i in range(1, 5):
    print('Final Validation accuracy for Uniform Initialization With Batch Size: %d is : %f' % (32 * i, acc_batch_uniform[i - 1][-1]))
for i in range(1, 5):
    print('Final Validation accuracy for Zeros Initialization With Batch Size: %d is : %f' % (32 * i, acc_batch_zeros[i - 1][-1]))

model = LogisticRegression("normal")

X_train_new = np.concatenate((X_train, X_valid), axis=0)
y_train_new = np.concatenate((y_train, y_valid), axis=0)
final_test_acc = model.fit(X_train_new, y_train_new,
                    X_test, y_test,
                    epochs=20,
                    batch_size=64,
                    learning_rate=1e-3)

# Final Test Accuracy For The Best Logistic Regression Model
x = list(range(20 + 1))
plt.figure(figsize=(18, 12))
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.xticks([x for x in range(0, 22, 2)])
plt.plot(x, final_test_acc, label='Multinomial Model')
plt.legend()
plt.title('Test Accuracy for the Logistic Regression Model')
plt.show()

y_preds = model.predict(X_test)
confusion = pd.crosstab(y_test, y_preds)

# Report the final test accuracy
print('Final Test Accuracy: %f' % (final_test_acc[-1]))

# print f1 score = 2 * precision * recall / (precision +  recall)
precision = confusion[1][1] / (confusion[0][1] + confusion[1][1])
recall = confusion[1][1] / (confusion[1][1] + confusion[1][0])
f1_score = (2 * precision * recall) / (precision + recall)
print("F1 Score For Final Logistic Regression Model: ", f1_score)

# plot confusion matrix
fig, ax = plt.subplots()
ax.matshow(confusion,cmap='OrRd')
ax.set(xlabel='Test', ylabel='Prediction')

for i in range(2):
  for j in range(2):
    c = confusion[j][i]
    ax.text(i, j, str(c), va='center', ha='center')
plt.title('Confusion Matrix For Final Logistic Regression Model')
plt.show()