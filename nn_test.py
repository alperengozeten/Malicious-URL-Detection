import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nn import NeuralNetwork
from itertools import product

df = pd.read_csv("data/url_processed.csv")
np.random.seed(2023)

# shuffle the data frame
df = df.sample(frac = 1)

# our feature matrix
X = df[['use_of_ip', 'url_length', 'subdomain_length', 'tld_length', 'fld_length', 'path_length',
       'count_letters', 'count_digits', 'count_puncs', 'count.', 'count@', 'count-',
       'count%', 'count?', 'count=',
       'letters_ratio', 'digit_ratio', 'punc_ratio', 'count_dirs',
       'use_of_shortener', 'first_dir_length',
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

nn_layers_list = [
    [64, 1],
    [32, 32, 1],
    [64, 64, 1],
]
nn_alphas = [1e-2, 5e-3, 1e-3, 5e-4]
nn_momentums = [0.85, 0.95]
nn_batch_sizes = [32, 64]

nn_hyperparams = list(product(nn_alphas, nn_momentums, nn_batch_sizes))

def plot_from_history(history, title):
    h = pd.DataFrame.from_dict(history)[['train_MSE', 'valid_MSE']]
    h = h.set_axis(['Train MSE', 'Test MSE'], axis=1)
    h.plot()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title(title)
    plt.show()

def subplot_mse(hist_list, nn_hyperparams, layers):
       fig, axs = plt.subplots(4, 4, figsize=(18, 12))
       for i in range(4):
              for j in range(4):
                     axs[i,j].set_title('Train and Validation MSE')
                     axs[i,j].set_xlabel('Epochs')
                     axs[i,j].set_ylabel('Validation MSE')
                     axs[i,j].plot(hist_list[4 * i + j], label='Batch size = 32')
                     axs[i,j].legend()

nn_scores = []
for layers in nn_layers_list:
     hist_list = []
     for alpha, momentum, batch_size in nn_hyperparams:
       nn = NeuralNetwork(n_neurons=layers)
       history = nn.fit(X_train, y_train, X_valid, y_valid, alpha=alpha, batch_size=batch_size, momentum=momentum, epochs=5, patience=5)
       hist_list.append(history)
     subplot_mse(hist_list, nn_hyperparams, layers)

best_model = NeuralNetwork(n_neurons=[64, 64, 1])
history = best_model.fit(X_train, y_train, X_test, y_test, alpha=0.005, batch_size=32, momentum=0.85,epochs=500, patience=5)

plot_from_history(history, 'MSE vs Epoch')