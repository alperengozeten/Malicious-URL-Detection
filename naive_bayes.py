import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

df = pd.read_csv("data/url_processed.csv")

# shuffle the data frame
df = df.sample(frac = 1)

# pick only the numeric features
X = df[['use_of_ip', 'url_length', 'subdomain_length', 'tld_length', 'fld_length', 'path_length',
       'count_letters', 'count_digits', 'count_puncs', 'count.', 'count@', 'count-',
       'count%', 'count?', 'count=', 'count_dirs', 'use_of_shortener', 'first_dir_length',
       'url_length_q', 'fld_length_q', 'https', 'count-https', 'count-http', 'count-www', 'sus_url']]

print(X.head())

y = df['is_malicious']

print(X.isna().sum())

# fill the nan values with 0
X = X.fillna(0)

# check again the number of nan values
print(X.isna().sum())

X.replace([np.inf, -np.inf], 0, inplace=True)

X['first_dir_length'] = X['first_dir_length'].astype('int64')
X['url_length_q'] = X['url_length_q'].astype('int64')
X['fld_length_q'] = X['fld_length_q'].astype('int64')

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

# convert df into numpy
x_train = X_train.to_numpy()
x_valid = X_valid.to_numpy()
x_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_valid = y_valid.to_numpy()

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

start_time = time.time()
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
print("--- %s seconds ---" % (time.time() - start_time))

multinomial_acc_history = []
def fit_smoothened_multinomial(alpha, spamFrequencies, normalFrequencies, x_train, x_test, y_test):
    regularizedSpamFrequencies = spamFrequencies + alpha
    regularizedNormalFrequencies = normalFrequencies + alpha

    # regularize the word counts for spam and normal categories, as well
    regularizedSpamWordCount = spamWordCount + alpha * x_train.shape[1]
    regularizedNormalWordCount = normalWordCount + alpha * x_train.shape[1]

    # divide by the total word count to find the probabilities
    regularizedSpamProbabilities = regularizedSpamFrequencies / regularizedSpamWordCount
    regularizedNormalProbabilities = regularizedNormalFrequencies / regularizedNormalWordCount

    # take the logarithm of the probabilities
    regularizedLogSpamProbabilities = np.log(regularizedSpamProbabilities)
    regularizedLogNormalProbabilities = np.log(regularizedNormalProbabilities)

    num_correct = tp = tn = fp = fn = 0
    for i in range(x_test.shape[0]):
        row = x_test[i]
        probSpam = np.log(p_spam)
        probNormal = np.log(p_normal)
        
        probSpam += (regularizedLogSpamProbabilities @ row)
        probNormal += (regularizedLogNormalProbabilities @ row)

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

    print("--------------- Multinomial Model With Additive Smoothing alpha = %d ---------------" % (alpha))
    print("The number of correct predictions: " + str(num_correct) + ", wrong predictions: " + str(x_test.shape[0] - num_correct) + ", accuracy: " + str(num_correct / x_test.shape[0]))
    print("The number of true positives: " + str(tp) + ", true negatives: " + str(tn))
    print("The number of false positives: " + str(fp) + ", false negatives: " + str(fn))
    multinomial_acc_history.append(num_correct / x_test.shape[0]) # add the accuracy to the history

# apply smoothing and compare on the validation set to pick the best smoothing
for i in range(0, 10, 2):
    fit_smoothened_multinomial(i, spamFrequencies, normalFrequencies, x_train, x_valid, y_valid)

# plotting the validation accuracy for different smoothing parameter values
x = list(range(0, 10, 2))
plt.figure(figsize=(18, 12))
plt.title('Smoothing Parameter Comparison')
plt.xlabel('Smoothing Parameter Alpha')
plt.ylabel('Validation accuracy')
plt.plot(x, multinomial_acc_history,  '-o', label='Multinomial Model')
plt.yticks([x / 5 for x in range(0, 6)])
for (parameter, acc) in zip(x, multinomial_acc_history):
    plt.text(parameter, acc, "{:.6f}".format(acc), va='bottom', ha='center')
plt.legend()
plt.title('Validation Accuracy For Multinomial Model')
plt.show()

# The best behaving multinomial model is one with smoothing = 0, report its accuracy again
# by training on the train + validation set 
x_train = np.concatenate((x_train, x_valid), axis=0)
y_train = np.concatenate((y_train, y_valid), axis=0)

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

num_correct = tp = tn = fp = fn = 0
y_pred = []
for i in range(x_test.shape[0]):
    row = x_test[i]
    probSpam = np.log(p_spam)
    probNormal = np.log(p_normal)
    
    probSpam += (logSpamProbabilities @ row)
    probNormal += (logNormalProbabilities @ row)

    predicted = 0
    if probSpam > probNormal:
        predicted = 1
    
    y_pred.append(predicted)
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
print("--------------- Multinomial Model ---------------")
print("The number of correct predictions: " + str(num_correct) + ", wrong predictions: " + str(x_test.shape[0] - num_correct) + ", accuracy: " + str(num_correct / x_test.shape[0]))
print("The number of true positives: " + str(tp) + ", true negatives: " + str(tn))
print("The number of false positives: " + str(fp) + ", false negatives: " + str(fn))

# plot confusion matrix
confusion = pd.crosstab(y_test, y_pred)
fig, ax = plt.subplots()
ax.matshow(confusion,cmap='OrRd')
ax.set(xlabel='Test', ylabel='Prediction')

for i in range(2):
  for j in range(2):
    c = confusion[j][i]
    ax.text(i, j, str(c), va='center', ha='center')
plt.title('Confusion Matrix For Multinomial Model With Smoothing = 0')
plt.show()

# print f1 score = 2 * precision * recall / (precision +  recall)
precision = tp / (fp + tp)
recall = tp / (tp + fn)
f1_score = (2 * precision * recall) / (precision + recall)
print("f1 score for multinomial model with smoothing = 0: ", f1_score)

"""
Bernoulli Model is trained on train + validation and its accuracy is reported
on the test dataset since it doesn't have any parameters to tune
"""
start_time = time.time()
def fit_bernoulli(x_train, x_test, y_train, y_test):
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

    print("--- %s seconds ---" % (time.time() - start_time))
    num_correct = tp = tn = fp = fn = 0
    y_pred_berno = []
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
        
        y_pred_berno.append(predicted)
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

    return y_pred_berno, tp, tn, fp, fn

x_train = X.iloc[:train_size].to_numpy()
x_valid = X.iloc[train_size: train_size + valid_size].to_numpy()
y_train = y.iloc[:train_size].to_numpy()
y_valid = y.iloc[train_size: train_size + valid_size].to_numpy()

# get a base accuracy score
y_pred_berno, tp, tn, fp, fn = fit_bernoulli(x_train, x_valid, y_train, y_valid)

# list of possible features
featureList = ['+', '#', '//', '$', '!', '*']
selectedFeatures = []
X_current = X.copy()
noIncrease = False
maxAcc = (tp + tn) / (tp + tn + fp + fn) # initial accuracy
print(maxAcc)

# Apply forward selection to get the best features
while not noIncrease:
    noIncrease = True
    maxChar = None
    for c in featureList:
        if c not in selectedFeatures:
            X_new = X_current.copy()
            X_new['count'+c] = df['count'+c]
            x_train_new = X_new.iloc[:train_size]
            x_valid_new = X_new.iloc[train_size: train_size + valid_size]
            y_pred_berno, tp, tn, fp, fn = fit_bernoulli(x_train_new, x_valid_new, y_train, y_valid)
            acc = (tp + tn) / (tp + tn + fp + fn)

            if acc > maxAcc:
                maxAcc = acc
                noIncrease = False
                maxChar = c
                print(maxAcc)
    
    if maxChar is not None:
        selectedFeatures.append(maxChar)
        X_current['count'+maxChar] = df['count'+maxChar]
        print(maxChar)

print('Selected Set Of Features:' + str(selectedFeatures))
print(X_current.head())

# The best behaving multinomial model is one with smoothing = 0, report its accuracy again
# by training on the train + validation set 
x_train = X_current.iloc[:train_size + valid_size].to_numpy()
x_test = X_current.iloc[train_size + valid_size:].to_numpy()
y_train = y.iloc[:train_size + valid_size].to_numpy()

y_pred_berno, tp, tn, fp, fn = fit_bernoulli(x_train, x_test, y_train, y_test)
print("--------------- Bernoulli Model ---------------")
print("The number of correct predictions: " + str(num_correct) + ", wrong predictions: " + str(x_test.shape[0] - num_correct) + ", accuracy: " + str((tp + tn) / x_test.shape[0]))
print("The number of true positives: " + str(tp) + ", true negatives: " + str(tn))
print("The number of false positives: " + str(fp) + ", false negatives: " + str(fn))

# plot confusion matrix
confusion = pd.crosstab(y_test, y_pred_berno)
fig, ax = plt.subplots()
ax.matshow(confusion,cmap='OrRd')
ax.set(xlabel='Test', ylabel='Prediction')

for i in range(2):
  for j in range(2):
    c = confusion[j][i]
    ax.text(i, j, str(c), va='center', ha='center')
plt.title('Confusion Matrix For Bernoulli Model')
plt.show()

# print f1 score = 2 * precision * recall / (precision +  recall)
precision = tp / (fp + tp)
recall = tp / (tp + fn)
f1_score = (2 * precision * recall) / (precision + recall)
print("f1 score for bernoulli model: ", f1_score)