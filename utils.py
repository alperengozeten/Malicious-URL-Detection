import re
import pandas as pd

from tld import get_tld
from typing import Tuple, Union
from urllib.parse import urlparse

def is_url_ip_address(url: str) -> bool:
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
    
def process_tld(url: str, fix_protos: bool = False) -> Tuple[str, str, str, str]:
    """
    Takes a URL string and uses the tld library to extract subdomain, domain, top
    level domain and full length domain
    """
    res = get_tld(url, as_object = True, fail_silently=False, fix_protocol=fix_protos)

    subdomain = res.subdomain
    domain = res.domain
    tld = res.tld
    fld = res.fld
        
    return subdomain, domain, tld, fld

def process_url_with_tld(row: pd.Series) -> Tuple[str, str, str, str]:
    """
    Takes in a dataframe row, checks to see if rows `is_ip` column is
    False. If it is false, continues to process the URL and extract the 
    features, otherwise sets four features to None before returning.
    
    This processing is wrapped in a try/except block to enable debugging
    and it prints out the inputs that caused a failure as well as a 
    failure counter.
    """
    try:
        if row['is_ip'] == 0:
            if str(row['url']).startswith('http:'):
                return process_tld(row['url'])
            else:
                return process_tld(row['url'], fix_protos=True)
        else:
            subdomain = None
            domain = None
            tld = None
            fld = None
            return subdomain, domain, tld, fld
    except:
        idx = row.name
        url = row['url']
        type = row['type']
        print(f'Failed - {idx}: {url} is a {type} example')
        return None, None, None, None

def get_url_path(url: str) -> Union[str, None]:
    """
    Get's the path from a URL
    
    For example:
    
    If the URL was "www.google.co.uk/my/great/path"
    
    The path returned would be "my/great/path"
    """
    try:
        res = get_tld(url, as_object = True, fail_silently=False, fix_protocol=True)
        if res.parsed_url.query:
            joined = res.parsed_url.path + res.parsed_url.query
            return joined
        else:
            return res.parsed_url.path
    except:
        return None
    
def alpha_count(url: str) -> int:
    """
    Counts the number of alpha characters in a URL
    """
    alpha = 0
    for i in url:
        if i.isalpha():
            alpha += 1
    return alpha

def digit_count(url: str) -> int:
    """
    Counts the number of digit characters in a URL
    """
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits

def count_dir_in_url_path(url_path: Union[str, None]) -> int:
    """
    Counts number of / in url path to count number of
    sub directories
    """
    if url_path:
        n_dirs = url_path.count('/')
        return n_dirs
    else:
        return 0

def get_first_dir_len(url_path: Union[str, None]) -> int:
    """
    Counts the length of the first directory within
    the URL provided
    """
    if url_path:
        if len(url_path.split('/')) > 1:
            first_dir_len = len(url_path.split('/')[1])
            return first_dir_len
    else:
        return 0
    
def contains_shortening_service(url: str) -> int:
    """
    Checks to see whether URL contains a shortening service
    """
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

def httpSecure(url):
    htp = urlparse(url).scheme
    match = str(htp)
    if match=='https':
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0