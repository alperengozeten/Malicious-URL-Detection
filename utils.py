import re
import pandas as pd

from tld import get_tld
from typing import Tuple, Union
from urllib.parse import urlparse

def contains_ip_address(url: str) -> bool:
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4 with port
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}|'
        '([0-9]+(?:\.[0-9]+){3}:[0-9]+)|'
        '((?:(?:\d|[01]?\d\d|2[0-4]\d|25[0-5])\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d|\d)(?:\/\d{1,2})?)', url)  # Ipv6
    if match:
        return 1
    else:
        return 0

""" returns full length domain (fld), top level domain (tld), domain, and
subdomain names for a given URL. These can be useful as features """  
def extract_tld(url: str, fix_protos: bool = False) -> Tuple[str, str, str, str]:
    res = get_tld(url, as_object = True, fail_silently=False, fix_protocol=fix_protos)

    fld = res.fld
    tld = res.tld
    domain = res.domain
    subdomain = res.subdomain
         
    return subdomain, domain, tld, fld

"""" takes an instance and checks if the url is an IP address. If yes, 
set four instances to None. If not, call extract_tld to process URL"""
def process_url(row: pd.Series) -> Tuple[str, str, str, str]:
    try:
        if row['use_of_ip'] == 0:
            if str(row['url']).startswith('http:'):
                return extract_tld(row['url'])
            else:
                return extract_tld(row['url'], fix_protos=True)
        else:
            fld = None
            tld = None
            domain = None
            subdomain = None
            return fld, tld, domain, subdomain
    except:
        index = row.name
        url = row['url']
        type = row['type']
        print(f'Fail: {index}: {url} is a type of: {type}') # in case processing fails, print failed instances and the count
        return None, None, None, None

""" returns the path of the url
for example: www.bilkent.com/srs/cgpa -> srs/cgpa """
def get_path(url: str) -> Union[str, None]:

    try:
        res = get_tld(url, as_object = True, fail_silently=False, fix_protocol=True)
        if res.parsed_url.query:
            joined = res.parsed_url.path + res.parsed_url.query
            return joined
        else:
            return res.parsed_url.path
    except:
        return None

""" return alpha characters count """
def alpha_count(url: str) -> int:

    a_count = 0
    for c in url:
        if c.isalpha():
            a_count += 1
    return a_count

""" return digits count """
def digit_count(url: str) -> int:

    d_count = 0
    for d in url:
        if d.isnumeric():
            d_count += 1
    return d_count

""" the number of '/' characters gives the number of subdirectories """
def sub_directory_count(url: Union[str, None]):
    if url:
        dir_count = url.count('/')
        return dir_count
    return 0

""" return length of the first directory """
def len_first_directory(url: Union[str, None]):
    if url:
        if len(url.split('/')) > 1:
            first_dir_len = len(url.split('/')[1])
            return first_dir_len
    return 0

""" check whether the URL has a shortening service or not
Shortening Service: a third-party website that converts that long URL to a short, 
case-sensitive alphanumeric code."""   
def check_shortening_service(url: str) -> int:

    check = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if check: return 1
    else: return 0

""" check whether the URL starts with 'http' or 'https' """
def check_http(url):
    htp = urlparse(url).scheme
    check = str(htp)

    if check == 'https':
        return 1
    else: return 0