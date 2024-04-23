from typing import List

import pandas as pd
import requests
import json
import re
from bs4 import BeautifulSoup
import datetime
import string
import random
import torch

def get_pipeline_prediction(image,image_processor, model, confidence_threshold=0.45, device='cpu'):
    with torch.no_grad():
        # load image and predict
        inputs = image_processor(images=image, return_tensors='pt').to(device)
        outputs = model(**inputs)

        # post-process
        target_sizes = torch.tensor([image.shape[:2]]).to(device)
        results = image_processor.post_process_object_detection(
            outputs=outputs,
            threshold=confidence_threshold,
            target_sizes=target_sizes
        )[0]
    return results


def get_dataframe(path: str, verbose:int=0):
    """
    Read a DataFrame from a csv file.

    Args:
        path (str): Path to the csv file.

    Returns:
        pd.DataFrame: Read DataFrame.
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        if verbose > 0:
            print(e)
        return pd.DataFrame()
def clean_string(s):
    if isinstance(s, str):
        return re.sub("[\W_]+",'',s)
    else:
        return s

def re_braces(s):
    if isinstance(s, str):
        return re.sub("[\(\[].*?[\)\]]", "", s)
    else:
        return s

def name_filter(s):
    s = clean_string(s)
    s = re_braces(s)
    s = str(s)
    s = s.replace(' ', '').lower()
    return s
def get_webpage(url):
    res = requests.get(url)
    if res.status_code == 200:
        return res.text
    return None


def get_webpage_soup(html, html_attr=None, attr_key_val=None):
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        if html_attr:
            soup = soup.find(html_attr, attr_key_val)
        return soup
    return None

def get_subpage_content(url:str):
    '''
    Scrape the content part of the sub page
    '''
    html = get_webpage(url)
    soup = get_webpage_soup(html)
    content = soup.find('div',{'id':'content'})
    return {
        'url':url,
        'content':str(content),
        'scraped_at':datetime.datetime.now()
    }

def image_renamer(name: str):
    return name.replace('-','_').replace(' ','')