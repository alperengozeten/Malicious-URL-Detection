import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nn import NeuralNetwork
from activation_functions import Linear, ReLU  

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

best_model = NeuralNetwork(n_neurons=[64, 64, 1])
history = best_model.fit(X_train, y_train, X_test, y_test, alpha=0.001, batch_size=32, momentum=0.85,epochs=50, patience=5)

h = pd.DataFrame.from_dict(history)[['train_MSE', 'valid_MSE']]
h = h.set_axis(['Train MSE', 'Test MSE'], axis=1)
h.plot()
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.title('MSE vs Epoch')
plt.show()