import pandas as pd
from utils import contains_ip_address, process_url, get_path, check_shortening_service, sub_directory_count, alpha_count, digit_count, len_first_directory, check_http
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("data/malicious_phish.csv")

print(df.head())

# create a column named is_ip by using the ip check function
df['use_of_ip'] = df['url'].apply(lambda i: contains_ip_address(i))

# get the number of URLs containing IP address
print(df['use_of_ip'].value_counts())

# an example url starting with the ip address
print(df[df['use_of_ip'] == 1].iloc[0]['url'])

print(df['url'][1])

# add new features
df[['subdomain', 'domain', 'tld', 'fld']] = df.apply(lambda x: process_url(x), axis=1, result_type="expand")

# General Features
df['url_path'] = df['url'].apply(lambda x: get_path(x))
df['contains_shortener'] = df['url'].apply(lambda x: check_shortening_service(x))

# URL component length
df['url_length'] = df['url'].apply(lambda x: len(str(x)))
df['subdomain_length'] = df['subdomain'].apply(lambda x: len(str(x)))
df['tld_length'] = df['tld'].apply(lambda x: len(str(x)))
df['fld_length'] = df['fld'].apply(lambda x: len(str(x)))
df['path_length'] = df['url_path'].apply(lambda x: len(str(x)))

# count the number of alphanumeric characters, digits, and punctuations
df['count_letters']= df['url'].apply(lambda i: alpha_count(i))
df['count_digits']= df['url'].apply(lambda i: digit_count(i))
df['count_puncs'] = (df['url_length'] - (df['count_letters'] + df['count_digits']))

# check special character counts for the url
for c in ".@-%?=":
    df['count'+c] = df['url'].apply(lambda a: a.count(c))
    # print(df['count'+c][:5])

df['count_dirs'] = df['url_path'].apply(lambda x: sub_directory_count(x))
df['first_dir_length'] = df['url_path'].apply(lambda x: len_first_directory(x))

# Binary Label by converting benign to 0 and all other classes to 1
df['is_malicious'] = df['type'].apply(lambda x: 0 if x == 'benign' else 1)

# Binned Features
groups = ['Short', 'Medium', 'Long', 'Very Long']
# URL Lengths in 4 bins
df['url_length_q'] = pd.qcut(df['url_length'], q=4, labels=groups)
# FLD Lengths in 4 bins
df['fld_length_q'] = pd.qcut(df['fld_length'], q=4, labels=groups)

# Counts for https, http, www
df['https'] = df['url'].apply(lambda i: check_http(i))
df['count-https'] = df['url'].apply(lambda a: a.count('https'))
df['count-http'] = df['url'].apply(lambda a: a.count('http'))
df['count-www'] = df['url'].apply(lambda a: a.count('www'))

# Percentage Features
df['letters_ratio'] = df['count_letters'] / df['url_length'] 
df['digit_ratio'] = df['count_digits'] / df['url_length']
df['punc_ratio'] = df['count_puncs'] / df['url_length']

enc = OrdinalEncoder()
df[["url_length_q","fld_length_q"]] = enc.fit_transform(df[["url_length_q","fld_length_q"]])

print(df.head())

# write back the file
df.to_csv('data/url_processed.csv', index=False)