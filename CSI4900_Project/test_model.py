import pandas as pd
import joblib
from urllib.parse import urlparse
from tld import get_tld
import re

# Load the trained model
model = joblib.load('rf_model.pkl')


def extract_features(url):
    try:
        domain = get_tld(url, as_object=True).fld

    except:

        domain = ''
    parsed_url = urlparse(url)
    features = {
        'url_len': len(url),
        'domain_len': len(domain if domain else ""),
        'https': 1 if parsed_url.scheme == 'https' else 0,
        'letters': sum(c.isalpha() for c in url),
        'digits': sum(c.isdigit() for c in url),
        '@': url.count('@'),
        '#': url.count('#'),
        '$': url.count('$'),
        '%': url.count('%'),
        '+': url.count('+'),
        '-': url.count('-'),
        '*': url.count('*'),
        '=': url.count('='),
        'comma': url.count(','),
        '.': url.count('.'),
        '?': url.count('?'),
        '!': url.count('!'),
        '//': url.count('//'),
        'url_shortened': int(bool(re.search(
            r"(bit\.ly|goo\.gl|tinyurl\.com|t\.co|ow\.ly|is\.gd|buff\.ly|adf\.ly|bit\.do)", url))),
        'contains_ip_address': int(bool(re.search(
            r'(\b25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?\b)(\.\b25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?\b){3}', url)))
    }
    return pd.DataFrame([features])


def test_urls(urls):
    for url in urls:
        features_df = extract_features(url)
        prediction = model.predict(features_df)
        print(f'URL: {url} - Prediction: {"Malicious" if prediction[0] == 1 else "Not Malicious"}')


if __name__ == "__main__":
    sample_urls = [
        "elegantanna.com.cn/img/?secure.runescape.com/m=weblogin/loginform.ws?mod=rxazdjvamp;ssl=0&dest",
        "https://secure.example.com",
        "http://malicious-site.com",
        "claudiacovafotos.com/wp-content/plugins/Update/login.php/",
        "https://bit.ly/3xyzABC",
        "http://192.168.1.1",
        "china-yhjt.com/css/?secure.runescape.com/m=weblogin/loginform.ws?mod=amp;tligpcvamp;ssl=0&amp;dest",
        "http://kxsnm.duckdns.org",
        "http://paul1623.hyperphp.com/",
        "e-mailupgradesdesk.webs.com/",
        "goodvibes.cl/server.php"
        "docs.google.com/forms/d/14fPjX1cGiAJdk8H7uOpLy_rRxGWyuBeGVe82BNYVhM4/viewform"

    ]
    test_urls(sample_urls)
