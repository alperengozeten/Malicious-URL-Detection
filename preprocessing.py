import pandas as pd
from utils import is_url_ip_address, process_url, get_path, check_shortening_service, sub_dir_count, alpha_count, digit_count, len_first_dir, check_http
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("data/malicious_phish.csv")

print(df.head())

# create a column named is_ip by using the ip check function
df['is_ip'] = df['url'].apply(lambda i: is_url_ip_address(i))

print(df['is_ip'].value_counts())

print(df['url'][1])

# add new features
df[['subdomain', 'domain', 'tld', 'fld']] = df.apply(lambda x: process_url(x), axis=1, result_type="expand")

# General Features
df['url_path'] = df['url'].apply(lambda x: get_path(x))
df['contains_shortener'] = df['url'].apply(lambda x: check_shortening_service(x))

# URL component length
df['url_len'] = df['url'].apply(lambda x: len(str(x)))
df['subdomain_len'] = df['subdomain'].apply(lambda x: len(str(x)))
df['tld_len'] = df['tld'].apply(lambda x: len(str(x)))
df['fld_len'] = df['fld'].apply(lambda x: len(str(x)))
df['url_path_len'] = df['url_path'].apply(lambda x: len(str(x)))

# count the number of alphanumeric characters, digits, and punctuations
df['url_alphas']= df['url'].apply(lambda i: alpha_count(i))
df['url_digits']= df['url'].apply(lambda i: digit_count(i))
df['url_puncs'] = (df['url_len'] - (df['url_alphas'] + df['url_digits']))

# check special character counts for the url
for c in ".@-%?=":
    df['count'+c] = df['url'].apply(lambda a: a.count(c))
    # print(df['count'+c][:5])

df['count_dirs'] = df['url_path'].apply(lambda x: sub_dir_count(x))
df['first_dir_len'] = df['url_path'].apply(lambda x: len_first_dir(x))

# Binary Label by converting benign to 0 and all other classes to 1
df['binary_label'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)

# Binned Features
groups = ['Short', 'Medium', 'Long', 'Very Long']
# URL Lengths in 4 bins
df['url_len_q'] = pd.qcut(df['url_len'], q=4, labels=groups)
# FLD Lengths in 4 bins
df['fld_len_q'] = pd.qcut(df['fld_len'], q=4, labels=groups)

# Counts for https, http, www
df['https'] = df['url'].apply(lambda i: check_http(i))
df['count-https'] = df['url'].apply(lambda a: a.count('https'))
df['count-http'] = df['url'].apply(lambda a: a.count('http'))
df['count-www'] = df['url'].apply(lambda a: a.count('www'))

# Percentage Features
df['pc_alphas'] = df['url_alphas'] / df['url_len'] 
df['pc_digits'] = df['url_digits'] / df['url_len']
df['pc_puncs'] = df['url_puncs'] / df['url_len']

enc = OrdinalEncoder()
df[["url_len_q","fld_len_q"]] = enc.fit_transform(df[["url_len_q","fld_len_q"]])

print(df.head())

# write back the file
df.to_csv('data/url_processed.csv', index=False)