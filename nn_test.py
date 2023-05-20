import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nn import NeuralNetwork
from itertools import product
from nn import calc_accuracy

df = pd.read_csv("data/url_processed.csv")
np.random.seed(2023)

# shuffle the data frame
df = df.sample(frac = 1)

# our feature matrix
X = df[['use_of_ip', 'url_length', 'subdomain_length', 'tld_length', 'fld_length', 'path_length',
       'count_letters', 'count_digits', 'count_puncs', 'count.', 'count@', 'count-',
       'count%', 'count?', 'count=', 'count+', 'count/', 'count,', 'count!',
       'letters_ratio', 'digit_ratio', 'punc_ratio', 'count_dirs',
       'use_of_shortener', 'first_dir_length',
       'url_length_q', 'fld_length_q', 'https', 'count-https', 'count-http', 'count-www', 'sus_url', 'is_php']]

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
    h = pd.DataFrame.from_dict(history)[['train_acc', 'valid_acc']]
    h = h.set_axis(['Train Acc', 'Test Acc'], axis=1)
    h.plot()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.show()

def subplot_mse(hist_list, nn_hyperparams, layers):

    fig, axs = plt.subplots(4, 4, figsize=(18, 18))
    for i in range(4):
        for j in range(4):
            axs[i,j].set_title('lr: ' + str(nn_hyperparams[4*i+j][0]) + ', mom: ' + str(nn_hyperparams[4*i+j][1]) + ', bs: ' + str(nn_hyperparams[4*i+j][2]), fontdict={'fontsize': 9})
            axs[i,j].set_xlabel('Epochs', fontdict={'fontsize': 8})
            axs[i,j].set_ylabel('Accuracy', fontdict={'fontsize': 8})
            axs[i,j].plot(hist_list[4 * i + j]['train_acc'], label='Train Acc')
            axs[i,j].plot(hist_list[4 * i + j]['test_acc'], label='Validation Acc')
            axs[i,j].legend(fontsize=5)
    plt.suptitle('Hyperparameter Tuning For NN With layers: ' + str(layers))
    plt.subplots_adjust(wspace=0.5, hspace=0.7)
    plt.show()

accList = []
for layers in nn_layers_list:
    hist_list = []
    for alpha, momentum, batch_size in nn_hyperparams:
        nn = NeuralNetwork(n_neurons=layers)
        history = nn.fit(X_train, y_train, X_valid, y_valid, alpha=alpha, batch_size=batch_size, momentum=momentum, epochs=200, patience=5)
        hist_list.append(history)
        accList.append(history['test_acc'])
    subplot_mse(hist_list, nn_hyperparams, layers)

accList = np.asarray(accList)
nn_best_index = np.argmax(accList)
nn_best_layer_index = nn_best_index // len(nn_hyperparams)
nn_best_layer = nn_layers_list[nn_best_layer_index]
print(nn_hyperparams[nn_best_index % len(nn_hyperparams)])

# train the best model on train + validation dataset, report metrics
# on the test dataset
X_train = X.iloc[:train_set_size + valid_set_size]
y_train = y.iloc[:train_set_size + valid_set_size]
best_model = NeuralNetwork(n_neurons=[64, 64, 1])
history = best_model.fit(X_train, y_train, X_test, y_test, alpha=0.005, batch_size=32, momentum=0.85,epochs=5, patience=5)

# plot the test and train accuries of the best model
plot_from_history(history, 'Accuracy vs Epoch For Best Model')

y_test = y_test.to_numpy()
y_preds = best_model.predict(X_test)
confusion = pd.crosstab(y_test.reshape((-1,)), y_preds.reshape((-1, )))

# Report the final test accuracy
print('Final Test Accuracy: %f' % (calc_accuracy(y_test.reshape((-1,)), y_preds.reshape((-1, )))))

# print f1 score = 2 * precision * recall / (precision +  recall)
precision = confusion[1][1] / (confusion[0][1] + confusion[1][1])
recall = confusion[1][1] / (confusion[1][1] + confusion[1][0])
f1_score = (2 * precision * recall) / (precision + recall)
print("F1 Score For Final Neural Network Model: ", f1_score)

# plot confusion matrix
fig, ax = plt.subplots()
ax.matshow(confusion,cmap='OrRd')
ax.set(xlabel='Test', ylabel='Prediction')

for i in range(2):
  for j in range(2):
    c = confusion[j][i]
    ax.text(i, j, str(c), va='center', ha='center')
plt.title('Confusion Matrix For Final Neural Network Model')
plt.show()