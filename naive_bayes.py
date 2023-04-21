import numpy as np
import pandas as pd

df = pd.read_csv("data/url_processed.csv")

# shuffle the data frame
df = df.sample(frac = 1)

X = df[['is_ip', 'url_len', 'subdomain_len', 'tld_len', 'fld_len', 'url_path_len',
       'url_alphas', 'url_digits', 'url_puncs', 'count.', 'count@', 'count-',
       'count%', 'count?', 'count=', 'count_dirs',
       'contains_shortener', 'first_dir_len',
       'url_len_q', 'fld_len_q']]

print(X.head())

y = df['binary_label']

print(X.isna().sum())

# fill the nan values with 0
X = X.fillna(0)

# check again the number of nan values
print(X.isna().sum())

X.replace([np.inf, -np.inf], 0, inplace=True)

X['first_dir_len'] = X['first_dir_len'].astype('int64')
X['url_len_q'] = X['url_len_q'].astype('int64')
X['fld_len_q'] = X['fld_len_q'].astype('int64')

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
print(X_train.dtypes)

# convert df into numpy
x_train = X_train.to_numpy()
x_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# calculate the spam and normal class probabilities
spam_count = np.count_nonzero(y_train)
normal_count = y_train.shape[0] - spam_count
p_spam = spam_count / y_train.shape[0]
p_normal = 1 - p_spam

# print the spam percentage
print("Spam percentage: " + str((p_spam)*100))

# add the number of words for each email
word_counts = np.sum(x_train, axis = 1)

# Get the total word counts for categories
spamWordCount = word_counts.T @ y_train
normalWordCount = word_counts.T @ (1 - y_train)

# Get the frequencies for each word seperately
spamFrequencies = y_train.T @ x_train
normalFrequencies = (1 - y_train.T) @ x_train 

# Calculate the parameters by dividing with the total word count 
spamProbabilities = spamFrequencies / spamWordCount
normalProbabilities = normalFrequencies / normalWordCount

# Take the log of the probabilities
with np.errstate(divide='ignore'):
    logSpamProbabilities = np.log(spamProbabilities)
    logNormalProbabilities = np.log(normalProbabilities)

logSpamProbabilities[np.isneginf(logSpamProbabilities)]= -1e+12
logNormalProbabilities[np.isneginf(logNormalProbabilities)]= -1e+12

# create a copy of the train and test datasets
bernoulli_x_train = np.copy(x_train)
bernoulli_x_test = np.copy(x_test)

# make nonzero positions 1 for the bernoulli model
bernoulli_x_train[bernoulli_x_train != 0] = 1
bernoulli_x_test[bernoulli_x_test != 0] = 1

# get the frequencies of the words for spam and normal categories
bernoulliSpamFrequencies = y_train.T @ bernoulli_x_train
bernoulliNormalFrequencies = (1 - y_train.T) @ bernoulli_x_train

bernoulliSpamProbabilities = bernoulliSpamFrequencies / spam_count
bernoulliNormalProbabilities = bernoulliNormalFrequencies / normal_count

num_correct = tp = tn = fp = fn = 0
for i in range(bernoulli_x_test.shape[0]):
    row = bernoulli_x_test[i]
    probSpam = np.log(p_spam)
    probNormal = np.log(p_normal)
    
    # calculate the spam probabilities for each word separately
    spamWordExists = (bernoulliSpamProbabilities * row)
    spamWordDoesntExist = ((1 - bernoulliSpamProbabilities) * (1 - row))
    spamWord = spamWordDoesntExist + spamWordExists
    with np.errstate(divide='ignore'):
        spamWord = np.log(spamWord)
    spamWord[np.isneginf(spamWord)] = -1e+12 # replace 0 with extremely small values  
    probSpam += np.sum(spamWord, axis=0)

    # calculate the normal probabilities for each word separately
    normalWordExists = (bernoulliNormalProbabilities * row)
    normalWordDoesntExist = ((1 - bernoulliNormalProbabilities) * (1 - row))
    normalWord = normalWordDoesntExist + normalWordExists
    with np.errstate(divide='ignore'):
        normalWord = np.log(normalWord)
    normalWord[np.isneginf(normalWord)] = -1e+12 # replace 0 with extremely small values  
    probNormal += np.sum(normalWord, axis=0)

    predicted = 0
    if probSpam > probNormal:
        predicted = 1
    
    if predicted == y_test[i]:
        num_correct += 1
        if predicted == 1:
            tp += 1
        else:
            tn += 1
    else:
        if predicted == 1:
            fp += 1
        else:
            fn += 1
print("--------------- Bernoulli Model ---------------")
print("The number of correct predictions: " + str(num_correct) + ", wrong predictions: " + str(x_test.shape[0] - num_correct) + ", accuracy: " + str(num_correct / x_test.shape[0]))
print("The number of true positives: " + str(tp) + ", true negatives: " + str(tn))
print("The number of false positives: " + str(fp) + ", false negatives: " + str(fn))