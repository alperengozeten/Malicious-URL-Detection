import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import is_url_ip_address, process_url_with_tld, get_url_path, contains_shortening_service
from tld import get_tld
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv("data/malicious_phish.csv")

print(df.head())

# create a column named is_ip by using the ip check function
df['is_ip'] = df['url'].apply(lambda i: is_url_ip_address(i))

print(df['is_ip'].value_counts())

print(df['url'][1])

df[['subdomain', 'domain', 'tld', 'fld']] = df.apply(lambda x: process_url_with_tld(x), axis=1, result_type="expand")

print(df.head())

# General Features
df['url_path'] = df['url'].apply(lambda x: get_url_path(x))
df['contains_shortener'] = df['url'].apply(lambda x: contains_shortening_service(x))

# URL component length
df['url_len'] = df['url'].apply(lambda x: len(str(x)))
df['subdomain_len'] = df['subdomain'].apply(lambda x: len(str(x)))
df['tld_len'] = df['tld'].apply(lambda x: len(str(x)))
df['fld_len'] = df['fld'].apply(lambda x: len(str(x)))
df['url_path_len'] = df['url_path'].apply(lambda x: len(str(x)))